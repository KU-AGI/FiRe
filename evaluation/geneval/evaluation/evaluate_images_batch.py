#!/usr/bin/env python3
"""
Evaluate generated images using Mask2Former + CLIP with Optimized Pipelining.
(Fully compatible with logic of the original serial script)
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import warnings
import threading
from queue import Queue
from typing import List, Dict, Any, Tuple
from functools import lru_cache

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import mmdet
from mmdet.apis import init_detector
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

import open_clip

# -----------------------------------------------------------------------------
# Globals / Constants
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "CUDA device is required."

ARGS = None
OBJECT_DETECTOR = None
CLIP_MODEL = None
TRANSFORM = None
TOKENIZER = None
CLASSNAMES = None
TEST_PIPELINE = None
TARGET_GPUS = None

# Pre-computed embeddings for zero-shot classification
CLASS_TEXT_EMBEDS = {}

FOLDERS_PER_PASS = None
NUM_WORKERS = None
PREFETCH_BATCHES = None
CLIP_BATCH_SIZE = None
CLIP_WORKERS = None

COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white"]

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Batch evaluator.")
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./")
    parser.add_argument("--options", nargs="*", type=str, default=[])
    
    # Optimization args
    parser.add_argument("--folders-per-pass", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        )
    return args

# -----------------------------------------------------------------------------
# Model Loading & Pre-computation
# -----------------------------------------------------------------------------
def load_models(args):
    print("[INFO] Loading Models...", file=sys.stderr)
    config_path = args.model_config
    detector_name = args.options.get("model", "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    ckpt_path = os.path.join(args.model_path, f"{detector_name}.pth")
    
    # Load Detector
    detector = init_detector(config_path, ckpt_path, device=DEVICE)

    # Load CLIP
    clip_arch = args.options.get("clip_model", "ViT-L-14")
    clip_model, _, transform = open_clip.create_model_and_transforms(
        clip_arch, pretrained="openai", device=DEVICE
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(clip_arch)

    # Load Class names
    names_path = os.path.join(os.path.dirname(__file__), "object_names.txt")
    with open(names_path) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return detector, (clip_model, transform, tokenizer), classnames

def precompute_all_class_embeddings(classnames):
    """
    Pre-computes CLIP text embeddings using the SAME templates as the original code.
    """
    global CLASS_TEXT_EMBEDS
    print("[INFO] Pre-computing text embeddings...", file=sys.stderr)
    
    # Exactly matching the original code's templates
    templates = [
        "a photo of a {} {}",          # (color, classname)
        "a photo of a {}-colored {}",  # (color, classname)
        "a photo of a {} object"       # (color)
    ]
    unique_classes = set(classnames)
    unique_classes.add("object") 

    with torch.no_grad():
        for cls_name in unique_classes:
            class_color_embeds = []
            for color in COLORS:
                texts = [
                    templates[0].format(color, cls_name),
                    templates[1].format(color, cls_name),
                    templates[2].format(color)
                ]
                tokens = TOKENIZER(texts).to(DEVICE)
                text_feats = CLIP_MODEL.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                # Mean embedding of 3 templates
                mean_feat = text_feats.mean(dim=0, keepdim=True)
                mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
                class_color_embeds.append(mean_feat)
            
            # [10, Dim] tensor (10 colors)
            CLASS_TEXT_EMBEDS[cls_name] = torch.cat(class_color_embeds, dim=0)

def build_test_pipeline(model):
    cfg = model.cfg.copy()
    test_pipeline = copy.deepcopy(cfg.data.test.pipeline)
    test_pipeline = replace_ImageToTensor(test_pipeline)
    return Compose(test_pipeline)

# -----------------------------------------------------------------------------
# 1. Streaming Detection Pipeline
# -----------------------------------------------------------------------------
def prepare_detector_input(filepath):
    return TEST_PIPELINE(
        dict(
            img_info=dict(filename=filepath),
            img_prefix=None,
            img=None,
            img_fields=[],
        )
    )

class DetectionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return prepare_detector_input(self.image_paths[idx])

def batched_inference_generator(tasks, batch_size):
    """
    Yields (subset_tasks, subset_detections) to allow interleaved CPU processing.
    """
    image_paths = [t["imagepath"] for t in tasks]
    dataset = DetectionDataset(image_paths)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        # Note: pin_memory=True with custom collate can sometimes cause issues in mmdet
        pin_memory=False, 
        collate_fn=lambda x: collate(x, samples_per_gpu=len(x)),
        persistent_workers=NUM_WORKERS > 0,
    )

    queue = Queue(maxsize=max(1, PREFETCH_BATCHES))

    def producer():
        for data in loader:
            queue.put(data)
        queue.put(None)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    current_idx = 0
    
    while True:
        data = queue.get()
        if data is None:
            break
        
        # Scatter unwraps DataContainer and moves to GPU
        if TARGET_GPUS:
            data = scatter(data, TARGET_GPUS, dim=0)[0]
            
        with torch.no_grad():
            # Mask2Former: return_loss=False returns (bbox, segm) usually
            batch_results = OBJECT_DETECTOR(return_loss=False, rescale=True, **data)
        
        batch_len = len(batch_results)
        yield tasks[current_idx : current_idx + batch_len], batch_results
        current_idx += batch_len

    prod_thread.join()

# -----------------------------------------------------------------------------
# 2. Logic & Evaluation (CPU)
# -----------------------------------------------------------------------------
def compute_iou(box_a, box_b):
    x_min, y_min = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x_max, y_max = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter_area = max(x_max - x_min + 1, 0) * max(y_max - y_min + 1, 0)
    area_a = max(box_a[2] - box_a[0] + 1, 0) * max(box_a[3] - box_a[1] + 1, 0)
    area_b = max(box_b[2] - box_b[0] + 1, 0) * max(box_b[3] - box_b[1] + 1, 0)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0

def extract_detections(bbox, segm, threshold, max_objects, nms_threshold):
    detected = {}
    for index, classname in enumerate(CLASSNAMES):
        cls_boxes = bbox[index]
        if cls_boxes is None or len(cls_boxes) == 0:
            continue
        # Sort by confidence
        ordering = np.argsort(cls_boxes[:, 4])[::-1]
        ordering = ordering[cls_boxes[ordering, 4] > threshold]
        ordering = ordering[:max_objects].tolist()
        
        kept = []
        while ordering:
            max_obj = ordering.pop(0)
            kept.append((cls_boxes[max_obj], None if segm is None else segm[index][max_obj]))
            # NMS
            ordering = [
                obj for obj in ordering
                if nms_threshold == 1 or compute_iou(cls_boxes[max_obj], cls_boxes[obj]) < nms_threshold
            ]
        if kept:
            detected[classname] = kept
    return detected

def relative_position(obj_a, obj_b, pos_threshold):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - pos_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    rels = set()
    if dx < -0.5: rels.add("left of")
    if dx > 0.5: rels.add("right of")
    if dy < -0.5: rels.add("above")
    if dy > 0.5: rels.add("below")
    return rels

def evaluate_detection(task, detection, result_index, opts):
    imagepath = task["imagepath"]
    metadata = task["metadata"]
    # Robust unpacking
    bbox, segm = detection if isinstance(detection, tuple) else (detection, None)
    if segm is None and isinstance(detection, tuple) and len(detection) > 1:
        segm = detection[1]

    threshold = opts['threshold'] if metadata.get("tag") != "counting" else opts['counting_threshold']
    detected = extract_detections(bbox, segm, threshold, opts['max_objects'], opts['nms_threshold'])

    result = {
        "filename": imagepath,
        "tag": metadata.get("tag", ""),
        "prompt": metadata.get("prompt", ""),
        "correct": True,
        "reasons": [],
        "metadata": json.dumps(metadata),
        "details": json.dumps({k: [box.tolist() for box, _ in v] for k, v in detected.items()}),
    }

    color_jobs = []
    matched_groups = []
    
    # [Include] Check
    for req_index, req in enumerate(metadata.get("include", [])):
        classname = req["class"]
        required = req.get("count", 1)
        available = detected.get(classname, [])
        
        # 1. Count Check
        if len(available) < required:
            result["correct"] = False
            result["reasons"].append(f"expected {classname}>={required}, found {len(available)}")
            matched_groups.append(None)
            continue

        # Take top-k confident objects (mimics original slicing)
        selected = available[:required]
        success = True

        # 2. Position Check
        # Note: We check position BEFORE color confirmation here (Parallelism trade-off).
        if "position" in req:
            expected_rel, target_idx = req["position"]
            if target_idx >= len(matched_groups) or matched_groups[target_idx] is None:
                result["correct"] = False
                result["reasons"].append(f"no target for {classname} to be {expected_rel}")
                success = False
            else:
                target_group = matched_groups[target_idx]
                for obj in selected:
                    match_found = False
                    for target_obj in target_group:
                        rels = relative_position(obj, target_obj, opts['position_threshold'])
                        if expected_rel in rels:
                            match_found = True
                            break 
                    if not match_found:
                        result["correct"] = False
                        true_rels = relative_position(obj, target_group[0], opts['position_threshold']) 
                        result["reasons"].append(
                             f"expected {classname} {expected_rel} target, found {' and '.join(true_rels)} target"
                        )
                        success = False
                        break
                        
        matched_groups.append(selected if success else None)

        # 3. Schedule Color Check (Deferred to GPU)
        if success and "color" in req:
            color_jobs.append({
                "result_index": result_index,
                "req_index": req_index,
                "imagepath": imagepath,
                "classname": classname,
                "objects": selected,
                "required_color": req["color"],
                "required_count": required,
            })

    # [Exclude] Check
    for req in metadata.get("exclude", []):
        classname = req["class"]
        limit = req.get("count", 1)
        detected_count = len(detected.get(classname, []))
        if detected_count >= limit:
            result["correct"] = False
            result["reasons"].append(f"expected {classname}<{limit}, found {detected_count}")

    return result, color_jobs

# -----------------------------------------------------------------------------
# 3. Color Classification (GPU) - CLIP Batch Processing
# -----------------------------------------------------------------------------
@lru_cache(maxsize=2000)
def load_image_cached(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

def apply_mask_and_crop(image, box, mask):
    x1, y1, x2, y2 = map(int, box[:4])
    
    # [기존 코드 논리 복원]
    # 1. 이미지를 먼저 자른다 (최적화 코드의 방식 - 결과 동일하지만 더 빠름)
    crop = image.crop((x1, y1, x2, y2))
    
    if mask is not None:
        
        mask_img = Image.fromarray(mask.astype(bool))
        
        if mask_img.size != image.size:
             mask_img = mask_img.resize(image.size, Image.NEAREST)
        
        mask_crop = mask_img.crop((x1, y1, x2, y2))
        
        bgcolor = ARGS.options.get("bgcolor", "#999")
        if bgcolor == "original":
            blank = crop.copy()
        else:
            blank = Image.new("RGB", crop.size, color=bgcolor)
            
        crop = Image.composite(crop, blank, mask_crop)
        
    return crop

class ColorJobDataset(Dataset):
    def __init__(self, jobs):
        self.jobs = jobs
        self.lookup = []
        for job_id, job in enumerate(jobs):
            for obj_id in range(len(job["objects"])):
                self.lookup.append((job_id, obj_id))

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        job_id, obj_id = self.lookup[idx]
        job = self.jobs[job_id]
        image = load_image_cached(job["imagepath"])
        box, mask = job["objects"][obj_id]
        crop = apply_mask_and_crop(image, box, mask)
        return TRANSFORM(crop), job_id

def run_color_jobs_batch(color_jobs):
    if not color_jobs: return {}

    dataset = ColorJobDataset(color_jobs)
    loader = DataLoader(
        dataset,
        batch_size=CLIP_BATCH_SIZE,
        shuffle=False,
        num_workers=CLIP_WORKERS,
        pin_memory=True,
        collate_fn=lambda b: (torch.stack([i[0] for i in b]), torch.tensor([i[1] for i in b], dtype=torch.long)),
        persistent_workers=CLIP_WORKERS > 0,
    )

    job_predictions = {idx: [] for idx in range(len(color_jobs))}
    
    with torch.no_grad():
        for images, job_ids in loader:
            images = images.to(DEVICE, non_blocking=True)
            
            img_feats = CLIP_MODEL.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            batch_text_embeds = []
            ids_list = job_ids.tolist()
            for jid in ids_list:
                classname = color_jobs[jid]["classname"]
                embeds = CLASS_TEXT_EMBEDS.get(classname, CLASS_TEXT_EMBEDS.get("object"))
                batch_text_embeds.append(embeds)
            
            batch_text_embeds = torch.stack(batch_text_embeds).to(DEVICE, non_blocking=True)
            
            # BMM: Image[B,1,D] x Text[B,D,10] -> Logits[B,1,10]
            logits = torch.bmm(img_feats.unsqueeze(1), batch_text_embeds.transpose(1, 2)).squeeze(1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            
            for job_id, pred in zip(ids_list, preds):
                job_predictions[job_id].append(pred)

    return job_predictions

def update_results_with_color(all_results, color_jobs, predictions):
    """Updates results based on deferred color checks."""
    for job_id, job in enumerate(color_jobs):
        result = all_results[job["result_index"]]
        preds = predictions.get(job_id, [])
        colors = [COLORS[idx] for idx in preds]
        
        count = colors.count(job["required_color"])
        if count < job["required_count"]:
            result["correct"] = False
            found_colors_str = ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
            result["reasons"].append(
                f"expected {job['required_color']} {job['classname']}>={job['required_count']}, found " +
                f"{count} {job['required_color']}; and {found_colors_str}"
            )

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def collect_tasks(dataset_root, subfolders, sample_regex):
    tasks = []
    for subfolder in subfolders:
        folderpath = os.path.join(dataset_root, subfolder)
        metadata_path = os.path.join(folderpath, "metadata.jsonl")
        if not os.path.isfile(metadata_path): continue
        with open(metadata_path) as fp: metadata = json.load(fp)
        samples_path = os.path.join(folderpath, "samples")
        if not os.path.isdir(samples_path): continue
        
        for imagename in os.listdir(samples_path):
            if sample_regex.match(imagename):
                imagepath = os.path.join(samples_path, imagename)
                if os.path.isfile(imagepath):
                    tasks.append({"imagepath": imagepath, "metadata": metadata, "folder": subfolder})
    return tasks

def main():
    dataset_root = os.path.abspath(ARGS.imagedir)
    print(f"[INFO] Processing dataset: {dataset_root}", file=sys.stderr)
    
    # Logic Options
    logic_opts = {
        'threshold': float(ARGS.options.get("threshold", 0.3)),
        'counting_threshold': float(ARGS.options.get("counting_threshold", 0.9)),
        'max_objects': int(ARGS.options.get("max_objects", 16)),
        'nms_threshold': float(ARGS.options.get("max_overlap", 1.0)),
        'position_threshold': float(ARGS.options.get("position_threshold", 0.1))
    }

    all_files = sorted(os.listdir(dataset_root))
    subfolders = [f for f in all_files if f.isdigit() and os.path.isdir(os.path.join(dataset_root, f))]
    sample_regex = re.compile(r"\d+\.png")
    
    dataset_results = []
    
    # Main Loop: Process in chunks if folders_per_pass is set
    chunk_size = FOLDERS_PER_PASS if FOLDERS_PER_PASS > 0 else len(subfolders)
    
    for start in tqdm(range(0, len(subfolders), chunk_size), desc="Processing folders"):
        group = subfolders[start : start + chunk_size]
        tasks = collect_tasks(dataset_root, group, sample_regex)
        if not tasks: continue
        
        #print(f"[INFO] Processing chunk {group[0]}-{group[-1]} ({len(tasks)} imgs)...", file=sys.stderr)
        
        pending_color_jobs = []
        batch_size = max(1, int(ARGS.options.get("det_batch", 32)))
        
        # Generator creates Detection stream
        gen = batched_inference_generator(tasks, batch_size)
        
        for batch_tasks, batch_detections in gen:
            # [CPU Phase] Logic Check
            for task, detection in zip(batch_tasks, batch_detections):
                res_idx = len(dataset_results)
                result, new_jobs = evaluate_detection(task, detection, res_idx, logic_opts)
                result["dataset"] = os.path.basename(dataset_root)
                dataset_results.append(result)
                pending_color_jobs.extend(new_jobs)
            
            # [GPU Phase] Interleaved Color Check
            if len(pending_color_jobs) >= CLIP_BATCH_SIZE * 2:
                preds = run_color_jobs_batch(pending_color_jobs)
                update_results_with_color(dataset_results, pending_color_jobs, preds)
                pending_color_jobs = []

        # Flush remaining
        if pending_color_jobs:
            preds = run_color_jobs_batch(pending_color_jobs)
            update_results_with_color(dataset_results, pending_color_jobs, preds)

    # Finalize (format reasons)
    final_output = []
    for res in dataset_results:
        reasons = res.pop("reasons", [])
        res["reason"] = "\n".join(reasons)
        final_output.append(res)

    if os.path.dirname(ARGS.outfile):
        os.makedirs(os.path.dirname(ARGS.outfile), exist_ok=True)
    with open(ARGS.outfile, "w") as fp:
        pd.DataFrame(final_output).to_json(fp, orient="records", lines=True)
    print(f"[INFO] Saved {len(final_output)} results to {ARGS.outfile}", file=sys.stderr)

if __name__ == "__main__":
    ARGS = parse_args()
    FOLDERS_PER_PASS = max(0, ARGS.folders_per_pass)
    NUM_WORKERS = max(0, ARGS.num_workers)
    PREFETCH_BATCHES = max(1, ARGS.prefetch_batches)
    CLIP_BATCH_SIZE = int(ARGS.options.get("clip_batch", 64))
    CLIP_WORKERS = int(ARGS.options.get("clip_workers", 4))

    OBJECT_DETECTOR, (CLIP_MODEL, TRANSFORM, TOKENIZER), CLASSNAMES = load_models(ARGS)
    precompute_all_class_embeddings(CLASSNAMES)
    TEST_PIPELINE = build_test_pipeline(OBJECT_DETECTOR)
    TARGET_GPUS = getattr(OBJECT_DETECTOR, "device_ids", [0])

    main()
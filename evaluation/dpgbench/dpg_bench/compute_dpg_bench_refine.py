import argparse
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument("--image-root-path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--csv", type=str, default="./dpg_bench/dpg_bench.csv")
    parser.add_argument("--res-path", type=str, default=None)
    parser.add_argument("--pic-num", type=int, default=1)
    parser.add_argument("--vqa-model", type=str, default="mplug")
    return parser.parse_args()


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt="damo/mplug_visual-question-answering_coco_large_en", device="gpu"):
        super().__init__()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline_vqa = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, image, question):
        input_vqa = {"image": image, "question": question}
        result = self.pipeline_vqa(input_vqa)
        return result["text"]


def prepare_dpg_data(args):
    previous_id = ""
    current_id = ""
    question_dict = dict()
    category_count = defaultdict(int)

    data = pd.read_csv(args.csv)
    for i, line in data.iterrows():
        if i == 0:
            continue

        current_id = line.item_id
        qid = int(line.proposition_id)

        dependency_list_str = line.dependency.split(",")
        dependency_list_int = []
        for d in dependency_list_str:
            dependency_list_int.append(int(d.strip()))

        if current_id == previous_id:
            question_dict[current_id]["qid2tuple"][qid] = line.tuple
            question_dict[current_id]["qid2dependency"][qid] = dependency_list_int
            question_dict[current_id]["qid2question"][qid] = line.question_natural_language
        else:
            question_dict[current_id] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language},
            )

        category = line.question_natural_language.split("(")[0].strip()
        category_count[category] += 1

        previous_id = current_id

    return question_dict


def crop_image(input_image, crop_tuple=None):
    if crop_tuple is None:
        return input_image
    return input_image.crop((crop_tuple[0], crop_tuple[1], crop_tuple[2], crop_tuple[3]))


def compute_dpg_one_sample(args, question_dict, image_path, vqa_model, resolution):
    generated_image = Image.open(image_path)

    # if resolution == 0, infer crop size from image size (assume 2x2 grid)
    if resolution == 0:
        width, _ = generated_image.size
        resolution = width // 2

    crop_tuples_list = [
        (0, 0, resolution, resolution),  # Top-left
        (resolution, 0, resolution * 2, resolution),  # Top-right
        (0, resolution, resolution, resolution * 2),  # Bottom-left
        (resolution, resolution, resolution * 2, resolution * 2),  # Bottom-right
    ]
    crop_tuples = crop_tuples_list[: args.pic_num]

    key = osp.basename(image_path).split(".")[0]
    value = question_dict.get(key, None)
    if value is None:
        raise KeyError(f"Missing item_id '{key}' in question_dict (check filenames vs CSV item_id).")

    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]

    scores = []
    per_crop_qid2scores = []
    per_crop_qid2scores_raw = [] 
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)

        # per-crop scoring
        qid2scores = dict()
        for qid, question in qid2question.items():
            answer = vqa_model.vqa(cropped_image, question)
            qid2scores[qid] = float(answer == "yes")

            # NOTE: multi-process appends can interleave; left as-is to match original behavior
            with open(args.res_path.replace(".txt", "_detail.txt"), "a") as f:
                f.write(image_path + ", " + str(crop_tuple) + ", " + question + ", " + answer + "\n")

        # qid2scores_orig = qid2scores.copy()
        per_crop_qid2scores_raw.append(qid2scores.copy())

        # apply dependency masking on a copy
        qid2scores_masked = qid2scores.copy()
        for qid, parent_ids in qid2dependency.items():
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores_masked.get(parent_id, 0.0) == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores_masked[qid] = 0.0

        per_crop_qid2scores.append(qid2scores_masked)

        score = sum(qid2scores_masked.values()) / len(qid2scores_masked)
        scores.append(score)

    average_score = sum(scores) / len(scores)

    qid2scores_agg = {}
    for qid in qid2question.keys():
        qid2scores_agg[qid] = float(np.mean([d[qid] for d in per_crop_qid2scores]))

    qid2scores_raw_agg = {}
    for qid in qid2question.keys():
        qid2scores_raw_agg[qid] = float(np.mean([d[qid] for d in per_crop_qid2scores_raw]))


    with open(args.res_path, "a") as f:
        f.write(image_path + ", " + ", ".join(str(i) for i in scores) + ", " + str(average_score) + "\n")

    return average_score, qid2tuple, qid2scores_agg, qid2scores_raw_agg


def main():
    args = parse_args()
    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    timestamp = time.time()
    time_style = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))

    if args.res_path is None:
        args.res_path = osp.join(args.image_root_path, f"dpg-bench_{time_style}_results.txt")

    if accelerator.is_main_process:
        with open(args.res_path, "w") as f:
            pass
        with open(args.res_path.replace(".txt", "_detail.txt"), "w") as f:
            pass

    device = str(accelerator.device)
    if args.vqa_model == "mplug":
        vqa_model = MPLUG(device=device)
    else:
        raise NotImplementedError(f"Unknown vqa_model: {args.vqa_model}")

    vqa_model = accelerator.prepare(vqa_model)
    vqa_model = getattr(vqa_model, "module", vqa_model)

    # 1) deterministic input list: images only + sorted
    filename_list = sorted(
        [
            f
            for f in os.listdir(args.image_root_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
    )

    # 2) safe shard split
    chunks = np.array_split(filename_list, accelerator.num_processes)
    local_filename_list = list(chunks[accelerator.process_index])

    local_scores = []
    local_category2scores = defaultdict(list)
    local_category2scores_raw = defaultdict(list)

    model_id = osp.basename(args.image_root_path)
    print(f"Start to conduct evaluation of {model_id}")

    for fn in tqdm(local_filename_list):
        image_path = osp.join(args.image_root_path, fn)
        try:
            score, qid2tuple, qid2scores, qid2scores_raw = compute_dpg_one_sample(
                args=args,
                question_dict=question_dict,
                image_path=image_path,
                vqa_model=vqa_model,
                resolution=args.resolution,
            )
            local_scores.append(score)

            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split("(")[0].strip()
                local_category2scores[category].append(float(qid2scores[qid]))
                local_category2scores_raw[category].append(float(qid2scores_raw[qid]))

        except Exception as e:
            print("Failed filename:", fn, e)
            continue

    accelerator.wait_for_everyone()

    # Global DPG score
    global_dpg_scores = gather_object(local_scores)
    mean_dpg_score = float(np.mean(global_dpg_scores)) if len(global_dpg_scores) > 0 else float("nan")

    # 3) category stats: gather once as (category, score) pairs
    local_pairs = []
    for category, scores in local_category2scores.items():
        for s in scores:
            local_pairs.append((category, float(s)))

    local_pairs_raw = []
    for category, scores in local_category2scores_raw.items():
        for s in scores:
            local_pairs_raw.append((category, float(s)))

    global_pairs = gather_object(local_pairs)
    global_pairs_raw = gather_object(local_pairs_raw)

    global_category2scores = defaultdict(list)
    global_category2scores_l1 = defaultdict(list)
    global_category2scores_raw = defaultdict(list)
    global_category2scores_l1_raw = defaultdict(list)
    global_categories = []

    if accelerator.is_main_process:
        for category, s in global_pairs:
            global_category2scores[category].append(s)
            l1_category = category.split("-")[0].strip()
            global_category2scores_l1[l1_category].append(s)

        global_categories = sorted(global_category2scores.keys())

        for category, s in global_pairs_raw:
            global_category2scores_raw[category].append(s)
            l1_category = category.split("-")[0].strip()
            global_category2scores_l1_raw[l1_category].append(s)

        print(f"=== Total evaluated images === {len(global_dpg_scores)}")

        print("=== Sample counts ===")
        for category in global_categories:
            scores = global_category2scores[category]
            print(f"{category}: {len(scores)} samples, mean: {np.mean(scores)}")
        print("====================")

        output = f"Model: {model_id}\n"

        output += "L1 category scores:\n"
        for l1_category in sorted(global_category2scores_l1.keys()):
            output += f"\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n"

        output += "L2 category scores:\n"
        for category in global_categories:
            output += f"\t{category}: {np.mean(global_category2scores[category]) * 100}\n"

        output += "\nL1 category scores (RAW, before dependency):\n"
        for l1_category in sorted(global_category2scores_l1_raw.keys()):
            output += f"\t{l1_category}: {np.mean(global_category2scores_l1_raw[l1_category]) * 100}\n"

        output += "L2 category scores (RAW, before dependency):\n"
        for category in global_categories:
            raw_scores = global_category2scores_raw.get(category, [])
            if len(raw_scores) == 0:
                output += f"\t{category}: nan\n"
            else:
                output += f"\t{category}: {np.mean(raw_scores) * 100}\n"

        output += f"Image path: {args.image_root_path}\n"
        output += f"Save results to: {args.res_path}\n"
        output += f"DPG-Bench score: {mean_dpg_score * 100}"

        with open(args.res_path, "a") as f:
            f.write(output + "\n")

        print(output)


if __name__ == "__main__":
    main()

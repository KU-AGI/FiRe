"""GenEval++ scoring using the FiRe experiment's GPT-4.1 evaluation protocol.

Dataset: https://github.com/yejy53/Echo-4o/tree/master/test_scripts
"""
import argparse
import base64
import concurrent.futures
import io
import json
import re
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
   - Object classes
   - Counts of each object
   - Colors of each object
   - Spatial position within the image (e.g., "above", "below", based on real pixel position)
   - Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
   - Do NOT allow stylized images (e.g., cartoons, sketches, anime).
   - Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment.

Return your result as a JSON object using this format:
{
  "correct": 1 if the image fully satisfies all expectations, else 0,
  "reason": "You may explain in detail what is missing or incorrect"
}
"""


def metadata_to_explanation(metadata):
    parts = []
    for item in metadata.get("include", []):
        obj, count = item["class"], item.get("count", 1)
        noun = f"{count} {obj + 's' if count > 1 else obj}"
        desc = []
        if item.get("color"):
            desc.append(f"{item['color']}-colored")
        if item.get("size"):
            desc.append(item["size"])
        if desc:
            noun = f"{' '.join(desc)} {noun}"
        region = item.get("region")
        text = f"{noun} located in the {region} part of the image" if region else f"{noun} present in the image"
        parts.append(f"- {text}.")
    for item in metadata.get("exclude", []):
        obj, count = item["class"], item.get("count", 1)
        parts.append(f"- No more than {count - 1} {obj + 's' if count > 1 else obj} should appear.")
    return "This image should contain:\n" + "\n".join(parts)


def extract_json_from_response(text):
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    result = json.loads(match.group(0))
    if result.get("correct") not in (0, 1):
        raise ValueError("Expected correct to be 0 or 1")
    return result


def evaluate_single_image(client, image_path, instruction, explanation):
    from PIL import Image

    if not image_path.exists():
        return {"correct": 0, "reason": f"Image not found: {image_path}"}
    with Image.open(image_path) as image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Instruction:\n{instruction}\n\nExpectation checklist:\n{explanation}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]},
    ]
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1", messages=messages, max_tokens=200, temperature=0.0
            )
            return extract_json_from_response(response.choices[0].message.content)
        except Exception as exc:
            if attempt == 3:
                return {"correct": -1, "reason": f"Failed after 3 retries: {exc}"}
            print(f"Retry {attempt} for {image_path}: {exc}")


def aggregate_results(results, strategy):
    valid = [r.get("correct", 0) for r in results if r.get("correct", 0) >= 0]
    if not valid:
        return 0.0
    if strategy == "any":
        return float(any(s == 1 for s in valid))
    if strategy == "all":
        return float(all(s == 1 for s in valid))
    if strategy == "majority":
        return float(sum(s == 1 for s in valid) >= len(valid) / 2)
    if strategy == "average":
        return sum(valid) / len(valid)
    raise ValueError(f"Unknown strategy: {strategy}")


def process_sample(client, idx, metadata, gen_dir, num_samples, strategy):
    instruction = metadata.get("prompt", "").strip()
    explanation = metadata_to_explanation(metadata)
    prompt_dir = gen_dir / f"{idx:05d}" / "samples"
    samples = [evaluate_single_image(client, prompt_dir / f"{s:05d}.png", instruction, explanation)
               for s in range(num_samples)]
    return {"id": idx, "tag": metadata.get("tag", ""), "prompt": instruction,
            "sample_results": samples, "aggregated_correct": aggregate_results(samples, strategy),
            "strategy": strategy}


def compute_stats(results):
    counts = defaultdict(lambda: {"correct": 0.0, "total": 0})
    for result in results:
        counts[result["tag"]]["total"] += 1
        counts[result["tag"]]["correct"] += result["aggregated_correct"]
    tags = {tag: round(c["correct"] / c["total"], 4) for tag, c in counts.items()}
    score = round(sum(tags.values()) / len(tags), 4) if tags else 0.0
    total = sum(c["total"] for c in counts.values())
    accuracy = round(sum(c["correct"] for c in counts.values()) / total, 4) if total else 0.0
    return tags, score, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-img-path", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["gen", "correction_0", "correction_1"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--strategy", choices=["average", "any", "all", "majority"], default="average")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    if args.num_samples < 1 or args.max_workers < 1:
        parser.error("num-samples and max-workers must be positive")

    import openai
    from tqdm import tqdm

    client = openai.OpenAI()  # Reads OPENAI_API_KEY from the environment.
    with args.metadata.open() as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    for stage in args.stages:
        gen_dir = args.base_img_path / stage
        if not gen_dir.is_dir():
            raise FileNotFoundError(f"Generation directory not found: {gen_dir}")
        missing = [gen_dir / f"{i:05d}" / "samples" / f"{s:05d}.png"
                   for i in range(len(metadata)) for s in range(args.num_samples)
                   if not (gen_dir / f"{i:05d}" / "samples" / f"{s:05d}.png").is_file()]
        print(f"[{stage}] {len(missing)} missing images (scored as 0).")
        for path in missing[:20]:
            print(f"  {path}")
        ckpt_path = args.base_img_path / f"eval_results_{stage}.jsonl"
        done = {}
        if ckpt_path.exists():
            with ckpt_path.open() as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        idx = record["id"]
                        if (idx < 0 or idx >= len(metadata)
                                or record["prompt"] != metadata[idx].get("prompt", "").strip()
                                or record["strategy"] != args.strategy
                                or len(record["sample_results"]) != args.num_samples):
                            raise ValueError(f"Checkpoint settings/data differ: {ckpt_path}. Use a fresh output directory.")
                        done[idx] = record
        pending = [(idx, meta) for idx, meta in enumerate(metadata) if idx not in done]
        print(f"[{stage}] Resuming: {len(done)}/{len(metadata)} prompts completed.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_sample, client, idx, meta, gen_dir, args.num_samples, args.strategy)
                       for idx, meta in pending]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=stage):
                result = future.result()
                # Only the main thread writes checkpoint records.
                with ckpt_path.open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                done[result["id"]] = result
        results = [done[idx] for idx in sorted(done)]
        tags, score, accuracy = compute_stats(results)
        output = {"eval_strategy": args.strategy, "num_samples_per_prompt": args.num_samples,
                  "overall_results": results, "tag_accuracy": tags,
                  "overall_score_mean_of_tags": score, "overall_accuracy_all_samples": accuracy}
        output_path = args.base_img_path / f"eval_results_{stage}.json"
        output_path.write_text(json.dumps(output, indent=2) + "\n")
        failures = sum(s["correct"] < 0 for r in results for s in r["sample_results"])
        print(f"[{stage}] Tag accuracy: {tags}")
        print(f"Mean of tags: {score:.2%}; all prompts: {accuracy:.2%}; API failures: {failures}")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

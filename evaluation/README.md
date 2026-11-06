# Evaluation

Install each benchmark's dependencies following its original repository, then run the commands below from the **FiRe repository root**. Adjust image and output paths to match your experiment. Replace `gen` with `correction_0`, `correction_1`, etc. to evaluate other stages.

## GenEval

**Setup:** Follow the [original GenEval installation instructions](https://github.com/djghosh13/geneval#setup).

Our version adds batch evaluation and saves per-task scores and incorrect cases.

```bash
bash evaluation/geneval/run_geneval.sh \
    ./outputs/fire/geneval/gen \
    ./outputs/fire/geneval/eval/gen \
    /path/to/mask2former \
    0
```

Arguments: image directory, output directory, model directory prepared during upstream setup, and GPU ID.

Results: `results.jsonl`, `score.json`, and `incorrect_cases.json` (when incorrect cases exist) in the output directory.

## GenEval++

**Setup:** See [Echo-4o's GenEval++ dataset and evaluation code](https://github.com/yejy53/Echo-4o/tree/master/test_scripts).

Our evaluator uses GPT-4.1, averages four image scores per prompt, and supports resuming from JSONL checkpoints.

```bash
pip install -r evaluation/geneval_pp/requirements.txt
export OPENAI_API_KEY="your-api-key"

python evaluation/geneval_pp/evaluate_geneval_pp.py \
    --metadata "/path/to/Echo-4o/test_scripts/Geneval++.jsonl" \
    --base-img-path ./outputs/fire/geneval_pp \
    --stages gen correction_0 correction_1
```

Use the same metadata file as inference. API evaluation incurs charges.

Results: `eval_results_<stage>.json` and `.jsonl` in the base image directory. The main score is `overall_score_mean_of_tags`. Rerun the same command to resume; use fresh result files if images or evaluation settings change.

## DPG-Bench

**Setup:** Follow the [original ELLA / DPG-Bench repository](https://github.com/TencentQQGYLab/ELLA#-dpg-bench) and its [evaluation requirements](https://github.com/TencentQQGYLab/ELLA/blob/main/requirements-for-dpg_bench.txt).

We address the following issues in the original evaluator:

- **Inconsistent ordering across GPUs:** Unsorted file and category ordering could cause inconsistent distributed evaluation. We sort input filenames, shard them deterministically, and gather category–score pairs before aggregation.
- **L1/L2 scores used only the last image:** Category scores retained answers from only the last of the four images per prompt. We average answers across all four images.
- **L1/L2 scores ignored dependencies:** Category scores did not account for question dependencies. We add dependency-aware L1/L2 scores and retain the original unmasked scores as `RAW` for comparison.

```bash
bash evaluation/dpgbench/run_dpgbench.sh \
    ./outputs/fire/dpgbench/final_outputs/gen/grid \
    0,1,2,3 \
    0
```

Arguments: 2×2 grid directory, GPU IDs (use `0` for one GPU), and tile resolution (`0` detects it automatically).

Results: `dpg-bench_<timestamp>_results.txt` and `dpg-bench_<timestamp>_results_detail.txt` in the grid directory. The main score is `DPG-Bench score`.

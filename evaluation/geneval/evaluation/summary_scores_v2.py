import argparse
import os
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

# --- [준비] 클래스 이름 로딩 ---
try:
    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]
        cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}
except FileNotFoundError:
    pass

# --- [준비] 결과 데이터 로딩 ---
# lines=True: JSONL 포맷 읽기
df = pd.read_json(args.filename, orient="records", lines=True)
results_data = {} 

# ==========================================
# 1. 화면 출력 & 데이터 집계
# ==========================================

# (1) Summary
total_images = len(df)
total_prompts = len(df.groupby('metadata'))
correct_images_pct = df['correct'].mean()
correct_prompts_pct = df.groupby('metadata')['correct'].any().mean()

results_data['summary'] = {
    'total_images': int(total_images),
    'total_prompts': int(total_prompts),
    'correct_images_percent': float(correct_images_pct),
    'correct_prompts_percent': float(correct_prompts_pct)
}

print("Summary")
print("=======")
print(f"Total images: {total_images}")
print(f"Total prompts: {total_prompts}")
print(f"% correct images: {correct_images_pct:.2%}")
print(f"% correct prompts: {correct_prompts_pct:.2%}")
print()

# (2) Task Breakdown
task_scores = []
results_data['tasks'] = {} 

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    mean_score = task_df['correct'].mean()
    correct_sum = task_df['correct'].sum()
    total_len = len(task_df)
    
    task_scores.append(mean_score)
    
    results_data['tasks'][tag] = {
        'score': float(mean_score),
        'correct': int(correct_sum),
        'total': int(total_len)
    }
    
    print(f"{tag:<16} = {mean_score:.2%} ({correct_sum} / {total_len})")
print()

overall_score = np.mean(task_scores) if task_scores else 0.0
results_data['overall_score'] = float(overall_score)
print(f"Overall score (avg. over tasks): {overall_score:.5f}")


# ==========================================
# 2. 파일 저장 (Score & Incorrect Cases)
# ==========================================
output_dir = os.path.dirname(os.path.abspath(args.filename))

# (1) Score 저장 (score.json)
score_output_path = os.path.join(output_dir, "score.json")
with open(score_output_path, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=4)
print(f"\nSaved scores to: {score_output_path}")

# (2) 오답 노트 저장 (incorrect_cases.json)
incorrect_df = df[df['correct'] == False].copy()

if not incorrect_df.empty:
    # A. JSON String -> Python Dict 변환 (가독성 향상)
    for col in ['metadata', 'details']:
        if col in incorrect_df.columns and len(incorrect_df) > 0:
            if isinstance(incorrect_df[col].iloc[0], str):
                try:
                    incorrect_df[col] = incorrect_df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except:
                    pass

    incorrect_output_path = os.path.join(output_dir, "incorrect_cases.json")
    
    # B. DataFrame -> List of Dicts 변환
    # (Pandas to_json 대신 Python built-in json을 쓰기 위함)
    incorrect_records = incorrect_df.to_dict(orient='records')

    # C. json.dump로 저장 (ensure_ascii=False로 한글 보호, indent=4로 줄바꿈)
    # Python 기본 json 라이브러리는 '/'를 이스케이프하지 않습니다.
    with open(incorrect_output_path, 'w', encoding='utf-8') as f:
        json.dump(incorrect_records, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(incorrect_df)} incorrect cases to: {incorrect_output_path}")
else:
    print("No incorrect cases found.")
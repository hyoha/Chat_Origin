from datasets import load_dataset
import json
import os

# ✅ 사용할 한국어 데이터셋
DATASET_NAME = "nlpai-lab/kullm-v2"
d
# ✅ 데이터 저장 경로 설정
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"🔹 데이터셋 다운로드 시작: {DATASET_NAME}")

# ✅ 한국어 데이터셋 로드
dataset = load_dataset(DATASET_NAME, split="train")

# ✅ 데이터셋 키 확인 (첫 번째 데이터 출력)
print(f"🔹 첫 번째 데이터 예제: {dataset[0]}")

# ✅ JSON 파일로 저장 (3000개 샘플 사용)
data_list = [{"question": item["instruction"], "answer": item["output"]} for item in dataset.select(range(3000))]

# ✅ 데이터 저장
dataset_path = os.path.join(DATA_DIR, "dataset.json")
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"✅ 데이터셋 다운로드 완료: {dataset_path}")

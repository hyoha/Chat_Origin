import os
import json

from torch._functorch._aot_autograd.logging_utils import model_name
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ 절대경로 설정
DATA_DIR = r"C:\Workspace\Chat_Origin\data"
RAW_FILE = os.path.join(DATA_DIR, "dataset.json")  # 원본 데이터
PREPROCESSED_PREFIX = os.path.join(DATA_DIR, "preprocessed_dataset")  # 분할 저장 경로

# ✅ 모델 선택
MODEL_NAME = "skt/kogpt2-base-v2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 모델의 모든 파라미터 확인
for name, param in model.named_parameters():
    print(f"파라미터 이름: {name}, 크기: {param.shape}")

# ✅ 패딩 토큰 강제 설정
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|startoftext|>", "<|sep|>", "<|endoftext|>"]})
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ✅ 최대 토큰 길이 및 분할 크기 설정
MAX_LENGTH = 128
SPLIT_SIZE = 30 * 1024 * 1024  # 🚀 30MB 단위로 파일을 나눔

def preprocess_data():
    """데이터셋을 토큰화하여 변환 후 여러 개의 JSON 파일로 나눠 저장"""
    print("🔄 한국어 챗봇 데이터 전처리 중...")

    # ✅ 원본 데이터 로드
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_data = []
    file_index = 1
    current_size = 0

    for example in raw_data:
        question = example["question"].strip()
        answer = example["answer"].strip()

        # ✅ 새롭게 데이터 포맷 설정
        formatted_text = f"<|startoftext|>{question}<|sep|>{answer}<|endoftext|>"

        # ✅ 토큰 변환
        tokens = tokenizer(
            formatted_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        data_entry = {
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist()
        }

        processed_data.append(data_entry)
        current_size += len(json.dumps(data_entry, ensure_ascii=False).encode("utf-8"))

        # ✅ 30MB를 넘으면 새로운 파일로 저장
        if current_size >= SPLIT_SIZE:
            output_file = f"{PREPROCESSED_PREFIX}_{file_index}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)

            print(f"✅ {output_file} 저장 완료 ({len(processed_data)}개 샘플)")
            processed_data = []
            current_size = 0
            file_index += 1

    # ✅ 남은 데이터 저장 (마지막 파일)
    if processed_data:
        output_file = f"{PREPROCESSED_PREFIX}_{file_index}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

        print(f"✅ {output_file} 저장 완료 ({len(processed_data)}개 샘플)")

    print("✅ 데이터 전처리 완료!")

if __name__ == "__main__":
    preprocess_data()

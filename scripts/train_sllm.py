import os
import json
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from tqdm import tqdm

# ✅ Windows `multiprocessing` 설정
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Windows 환경에서 multiprocessing 오류 방지

    # ✅ 경로 설정
    MODEL_NAME = "skt/kogpt2-base-v2"
    MODEL_SAVE_PATH = r"C:\Workspace\Chat_Origin\models\finetuned_model"
    DATA_DIR = r"C:\Workspace\Chat_Origin\data"
    DATA_PREFIX = os.path.join(DATA_DIR, "preprocessed_dataset")

    # ✅ 모델 및 토크나이저 로드
    print(f"🔹 모델 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ✅ 특수 토큰 추가
    special_tokens = {"additional_special_tokens": ["<|startoftext|>", "<|sep|>", "<|endoftext|>"]}
    tokenizer.add_special_tokens(special_tokens)

    # ✅ 모델 로드 후 토큰 크기 조정
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))  # ✅ 새로운 토큰을 반영하여 모델 재조정

    # ✅ 모델의 단어 사전 크기 확인
    vocab_size = model.config.vocab_size
    print(f"🔹 모델 vocab_size: {vocab_size}")

    # ✅ 데이터 로드 (여러 개의 분할된 데이터 파일 자동 로드)
    print("🔄 학습 데이터 로드 중...")
    dataset = []
    file_index = 1

    while True:
        data_file = f"{DATA_PREFIX}_{file_index}.json"
        if not os.path.exists(data_file):
            break

        with open(data_file, "r", encoding="utf-8") as f:
            dataset.extend(json.load(f))

        print(f"✅ {data_file} 로드 완료 ({len(dataset)}개 샘플)")
        file_index += 1

    # ✅ 데이터 개수 확인
    print(f"🔹 총 학습 샘플 개수: {len(dataset)}개")


    # ✅ PyTorch Dataset 변환
    class CustomDataset(Dataset):
        def __init__(self, data, vocab_size):
            self.data = data
            self.vocab_size = vocab_size

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # ✅ input_ids 값이 vocab_size를 초과하지 않도록 필터링
            input_ids = [min(token, self.vocab_size - 1) for token in item["input_ids"]]

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long)
            }


    train_dataset = CustomDataset(dataset, vocab_size)

    # ✅ Windows 최적화 설정 (멀티프로세싱 이슈 해결)
    BATCH_SIZE = 2  # ✅ 배치 크기 증가
    EPOCHS = 3  # ✅ 학습 횟수 증가
    GRADIENT_ACCUMULATION_STEPS = 4  # ✅ 메모리 사용 최적화
    LEARNING_RATE = 2e-4  # ✅ CPU 학습 최적화
    NUM_WORKERS = 0  # ✅ Windows에서는 0으로 설정하여 멀티프로세싱 문제 방지

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=False
    )

    # ✅ 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS,
    )

    # ✅ 학습 환경 설정 (CPU 사용)
    device = torch.device("cpu")
    model.to(device)

    # ✅ 학습 루프 실행
    print("🔹 모델 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        with tqdm(total=len(train_loader), desc=f"🚀 Epoch {epoch + 1}/{EPOCHS}") as pbar:
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        print(f"✅ Epoch {epoch + 1}/{EPOCHS} 완료 - 평균 Loss: {total_loss / len(train_loader):.4f}")

    # ✅ 모델 저장
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print(f"✅ 모델 학습 완료 & 저장됨: {MODEL_SAVE_PATH}")

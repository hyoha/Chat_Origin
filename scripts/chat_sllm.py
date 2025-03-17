import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fuzzywuzzy import process  # 🔹 유사도 기반 검색을 위한 라이브러리

# ✅ 경로 설정
MODEL_PATH = r"C:\Workspace\Chat_Origin\models\finetuned_model"
DATA_PATH = r"C:\Workspace\Chat_Origin\data\data.txt"

# ✅ 모델 및 토크나이저 로드
print("🔹 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# ✅ 특수 토큰 추가 (반드시 모델과 일치해야 함)
special_tokens = {"additional_special_tokens": ["<|startoftext|>", "<|sep|>", "<|endoftext|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cpu")
model.to(device).eval()


# ✅ data.txt 파일 로드 및 분석
def load_data():
    """ data.txt 파일을 로드하여 사전 형태로 저장 """
    if not os.path.exists(DATA_PATH):
        print("⚠️ data.txt 파일이 없습니다. 참고 데이터 없이 실행됩니다.")
        return {}

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_dict = {}
    for line in lines:
        if ":" in line:  # 질문과 답변을 구분 (예: "질문: 답변")
            parts = line.split(":", 1)
            question, answer = parts[0].strip(), parts[1].strip()
            data_dict[question] = answer

    print(f"✅ {len(data_dict)}개의 데이터 로드 완료!")
    return data_dict


# ✅ 데이터 로드
data_dict = load_data()


def find_best_match(user_input):
    """ 사용자의 질문과 가장 유사한 질문을 data.txt에서 찾아서 반환 """
    if not data_dict:
        return None  # 데이터가 없으면 패스

    best_match, score = process.extractOne(user_input, data_dict.keys())

    if score > 75:  # 🔹 유사도 75% 이상이면 참고 답변으로 제공
        return data_dict[best_match]

    return None


def clean_response(response):
    """불필요한 특수 토큰 제거"""
    response = response.replace("<|startoftext|>", "").replace("<|sep|>", "").replace("<|endoftext|>", "")
    response = response.replace("<|endoftext|>", "").strip()
    return response


def chat():
    print("💬 한국어 챗봇 실행 중... 종료하려면 'exit' 입력")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # ✅ data.txt에서 가장 유사한 질문 찾기
        reference_answer = find_best_match(user_input)

        # ✅ 모델 입력 생성
        input_text = f"<|startoftext|>{user_input}<|sep|>"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                temperature=0.7
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = clean_response(response)

        # ✅ 참고할 답변이 있으면 같이 제공
        if reference_answer:
            print(f"🔹 참고 답변: {reference_answer}")
            print(f"🤖 모델 답변: {response}")
        else:
            print(f"🤖 Bot: {response}")


if __name__ == "__main__":
    chat()

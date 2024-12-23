import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


#############################
# Settings
#############################
base_model_name = "openai-community/gpt2"
lora_model_path = "lora-checkpoints/final"
max_new_tokens = 100
input_to_chat_template = "<user>:{user_input}\n<assistant>:"


def load_lora_model(base_model_name: str, lora_model_path: str):
    """
    1) base_model 로드
    2) LoRA 가중치(lora_model_path) 로드
    3) eval 모드
    """
    # 1) 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # 2) LoRA 가중치 로드
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # 3) 추론 모드
    lora_model.eval()
    return lora_model


def generate_response(model, tokenizer, user_input: str, max_new_tokens: int):
    """
    LoRA 모델에 user_input을 주고 결과 텍스트를 반환
    """
    # training 때 사용했던 prompt 형식 맞추기
    prompt = input_to_chat_template.format(user_input=user_input)

    # 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()

    # 생성
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 디코딩
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # "<assistant>:" 부분 뒤에만 잘라서 반환할 수도 있음
    # 여기선 편의상 전체 문장을 그대로 반환
    return generated_text


def main():
    # 1) 토크나이저, 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_lora_model(base_model_name, lora_model_path)

    # 2) 간단 테스트: 사용자 입력 반복
    print("===== LoRA Inference =====")
    while True:
        user_input = input("user> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        # 응답 생성
        response = generate_response(model, tokenizer, user_input, max_new_tokens=max_new_tokens)
        print(f"assistant> {response}\n")


if __name__ == "__main__":
    main()

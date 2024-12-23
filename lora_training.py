"""
lora_training.py

result.jsonl (dataset_generation.py 에서 생성됨) 파일에서
success=true 인 데이터만 불러와서 모델 훈련 (LoRA)
"""

import json
import torch
from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


#############################
# Run Settings
#############################
result_path = "result.jsonl"
base_model = "openai-community/gpt2"
target_modules = ["c_attn"]
input_target_join_fn = lambda inp, tgt: "<user>:" + inp + "\n" + "<assistant>:" + tgt
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.05


def load_success_data_from_jsonl(result_path: str) -> List[Dict[str, str]]:
    """
    result.jsonl을 순회하여,
    success=true 인 레코드만 (question, planning) 추출
    -> (input_text, target_text) 구조로 변환
    """
    training_data = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except:
                continue

            if not row.get("success"):
                # success가 False거나 없으면 skip
                continue

            question = row.get("question", "")
            planning_str = row.get("planning", "")

            # dataset_generation.py 의 prompt와 동일하게 맞춤
            input_text = f"""\
Please write a list of *search queries* to solve the following question.
Return only the JSON string (no additional text).

The search queries must be written in **the same language** as the question.

question: {question}

Example:
Question: What is the difference between a cat and a dog?

{{
  "search_queries": [
    "cat definition", 
    "dog definition",
    "differences between a cat and a dog"
  ]
}}
"""
            target_text = planning_str

            training_data.append({"input_text": input_text, "target_text": target_text})
    return training_data


def train_lora_model(
    training_list: List[Dict[str, str]],
    base_model_name: str,
    output_dir: str,
):
    raw_dataset = Dataset.from_list(training_list)
    # 데이터셋을 train/eval로 분할 (90%/10%)
    split_dataset = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias="none"
    )
    model = get_peft_model(base_model, peft_config)

    def data_collator(batch):
        input_texts = [ex["input_text"] for ex in batch]
        target_texts = [ex["target_text"] for ex in batch]
        joined_texts = [input_target_join_fn(inp, tgt) for inp, tgt in zip(input_texts, target_texts)]
        encoded = tokenizer(joined_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final")


def main():
    """
    python lora_training.py 실행 시:
    - result.jsonl에서 success=true인 데이터만 로드
    - LoRA 훈련
    """
    # 1) result.jsonl -> (input_text, target_text) 변환
    training_list = load_success_data_from_jsonl(result_path)
    print(f"훈련 데이터 개수: {len(training_list)}")

    # 2) LoRA 훈련
    train_lora_model(training_list=training_list, base_model_name=base_model, output_dir="lora-checkpoints")

    print("LoRA 훈련 완료. 모델이 lora-checkpoints/final 에 저장되었습니다.")


if __name__ == "__main__":
    main()

"""
dataset_generation.py

- Natural Questions 데이터셋을 streaming=True 로 불러옴
- LLM을 호출하여 question -> planning(JSON) 생성
- 결과를 result.jsonl 에 기록 (success/fail 통합)
- 이미 생성된 result.jsonl 이 있다면 마지막 인덱스+1부터 이어서 처리
"""

import json
import os
from typing import Dict, Any, Optional
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv
import time

#############################
# Run Settings
#############################
# dataset_name = "google-research-datasets/natural_questions"
# question_text_position = ["question", "text"]
# result_path = "google-research-datasets-natural-questions.jsonl"
# dataset_config = "default"
# split = "train"
# sleep_time = None

dataset_name = "KETI-AIR/kor_squad_v2"
question_text_position = ["question"]
result_path = "KETI-AIR-kor-squad-v2.jsonl"
dataset_config = None
split = "train"
sleep_time = 0.1

#############################
# OpenAI Settings
#############################
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_host = os.getenv("OPENAI_HOST")
model_name = os.getenv("MODEL_NAME")
client = OpenAI(api_key=openai_api_key, base_url=openai_host)


#############################
# Functions
#############################
def _load_streaming_dataset(
    dataset_name: str = dataset_name,
    dataset_config: Optional[str] = dataset_config,
    split: str = split,
):
    """
    streaming=True 로 dataset을 로드하여,
    한 번에 다운로드하지 않고 순회(이터레이터) 방식으로 접근 가능하게 한다.
    """
    return load_dataset(dataset_name, dataset_config, split=split, streaming=True)


def _generate_planning(
    question: str,
    client: OpenAI,
    model_name: str,
) -> str:
    """
    [Atom]
    단일 question에 대해 LLM을 호출,
    '단계별 해결 과정을 담은 JSON' 문자열을 만들어 반환.
    """
    prompt = f"""\
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
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": "{"}],
    )
    content = response.choices[0].message.content
    content = "{" + content
    return content


def _process_single_question(
    idx: int,
    question: str,
    client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    """
    [Molecule]
    1. _generate_planning 호출
    2. 반환된 JSON 문자열이 valid한지 확인
    3. 성공이면 { index, question, planning, success=True }
       실패면  { index, question, error, success=False }
    """
    try:
        planning_str = _generate_planning(
            question=question,
            client=client,
            model_name=model_name,
        )
        # JSON 파싱이 유효한지 검사
        json.loads(planning_str)

        if sleep_time:
            time.sleep(sleep_time)

        return {
            "index": idx,
            "question": question,
            "success": True,
            "planning": planning_str,
        }
    except Exception as e:
        # 실패한 경우
        return {
            "index": idx,
            "question": question,
            "success": False,
            "error": f"{type(e).__name__}: {planning_str}",
        }


#############################
# Organism Functions
#############################
def _get_last_index(result_path: str) -> int:
    """
    result.jsonl 파일이 이미 있다면,
    가장 마지막 인덱스(max index)를 찾아서 반환.
    파일이 없거나, 아무 기록도 없으면 -1 반환.
    """
    if not os.path.exists(result_path):
        return -1

    last_idx = -1
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                idx_val = row.get("index", -1)
                if idx_val > last_idx:
                    last_idx = idx_val
            except:
                pass
    return last_idx


def generate_result_jsonl(
    client: OpenAI,
    model_name: str,
    result_path: str,
    count: int,
):
    """
    [Organism]
    - 이미 result.jsonl가 있으면 마지막 index를 찾고, 그 다음부터 처리
    - NaturalQuestions에서 streaming으로 질문을 읽고, (count)개만 처리
    - 처리 결과를 result.jsonl 에 저장
    """
    # 1) 현재까지의 last_index 확인
    last_idx = _get_last_index(result_path)
    start_index = last_idx + 1  # 다음부터 처리

    dataset = _load_streaming_dataset()  # streaming load

    # 2) 결과 파일 (append 모드)
    f_out = open(result_path, "a", encoding="utf-8")

    num_processed = 0
    try:
        for i, sample in enumerate(dataset):
            # i < start_index 까지는 skip
            if i < start_index:
                continue
            # count개 처리했으면 중단
            if num_processed >= count:
                print(f"[Info] 목표 {count}개 생성, 종료.")
                break

            question_text = sample
            for pos in question_text_position:
                question_text = question_text[pos]

            result = _process_single_question(
                idx=i,
                question=question_text,
                client=client,
                model_name=model_name,
            )

            # result.jsonl 에 기록
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            num_processed += 1

        print(f"[Info] {num_processed}개 생성 완료.")
    finally:
        f_out.close()


def check_is_alive(client: OpenAI, model_name: str):
    """
    LLM 서버 정상 응답 여부 간단 체크
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello?"}],
        )
        print("Alive Check OK")
        print("Sample response:", response.choices[0].message.content)
    except Exception as e:
        print("[Error in check_is_alive]", e)
        raise


def main():
    """
    [Page]
    예: python dataset_generation.py 실행 시,
    - result.jsonl 파일을 확인해서 이미 있는 마지막 인덱스+1부터
    """
    try:
        check_is_alive(client, model_name=model_name)
    except Exception as e:
        print(f"[Error] {e}")
        return

    generate_result_jsonl(
        client=client,
        model_name=model_name,
        result_path=result_path,
        count=1000,
    )


if __name__ == "__main__":
    main()

import json


#############################
# Run Settings
#############################
datasets_path = ["google-research-datasets-natural-questions.jsonl", "KETI-AIR-kor-squad-v2.jsonl"]
result_path = "result.jsonl"


def main():
    result_dataset = []
    for dataset_path in datasets_path:
        with open(dataset_path, "r") as f:
            for line in f:
                record = json.loads(line)
                result_dataset.append(record)

    with open(result_path, "w") as f:
        for record in result_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

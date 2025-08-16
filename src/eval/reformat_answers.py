import os

from tqdm import tqdm

from src.arguments import parse_args
from src.utils import load_dataset_for_eval, read_jsonl, write_jsonl


def main(args):
    dataset = load_dataset_for_eval(args)

    for root, _, files in os.walk("outputs/answers_in_short_forms"):
        for file in files:
            if os.path.basename(args.dataset_config).split(".")[0] not in file:
                continue

            print(file)

            answers = {}
            for index_row, row in enumerate(tqdm(dataset, desc=f"[Eval] {args.dataset_name.split('/')[-1]}")):
                id, example_id = str(row['id']), str(row['example_id'])
                if id not in answers:
                    answers[id] = {}
                answers[id][example_id] = row['answer']

            generated_results = read_jsonl(os.path.join(root, file))
            generated_results_new = []

            for index_row, row in enumerate(tqdm(generated_results, desc=f"[Eval] {args.dataset_name.split('/')[-1]}")):
                id, example_id = str(row['id']), str(row['question_id'])
                if id not in answers:
                    continue
                if example_id not in answers[id]:
                    continue

                generated_results_new.append({
                    "id": id,
                    "question_id": example_id,
                    "question": row['question'],
                    "pred": row['pred'],
                    "answer": answers[id][example_id],
                })
            write_jsonl(generated_results_new, os.path.join(args.output_dir, os.path.basename(root), file))


if __name__ == "__main__":
    args = parse_args('metrics')
    main(args)

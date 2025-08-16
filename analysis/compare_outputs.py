import os

from utils.data_utils import read_jsonl
from utils.utility import print_colored

output_file_name1 = "outputs/mistral-7b-instruct-v0.2/answers_QASPER.jsonl"
output_file_name2 = "outputs/Mistral7B_20M_paragraph3M_SQuAD/answers_QASPER_simplified.jsonl"


generated_results_1 = read_jsonl(output_file_name1)
generated_results_2 = read_jsonl(output_file_name2)

assert len(generated_results_1) == len(generated_results_2)

for i in range(10):
    pred1 = generated_results_1[i]['predict'].replace("<unk>", "").replace("<pad>", "").strip()
    pred2 = generated_results_2[i]['predict'].replace("<unk>", "").replace("<pad>", "").strip()

    print('=' * 15)
    print(generated_results_1[i]['question'])
    print_colored(generated_results_1[i]['answer'], "red")
    print_colored(pred1, "yellow")
    print_colored(pred2, "blue")


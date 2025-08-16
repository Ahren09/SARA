import json
import os
import sys

import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
# third party
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

from src.utils.data_utils import keyword_extraction_with_tfidf


# built-in

from src.metrics.metrics import eval_fact_checking
from src.arguments import parse_args
from src.utils.eval_utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_truthfulqa,
    prepare_retrieval_embeds,
    create_prompt_with_mistral_chat_format,
    get_start_prompt,
)

from src.utils.data_utils import reformat_prompt_qasper, read_jsonl, write_jsonl

from src.const import COMPRESS
from src.model import (
    SFR,
)


QA_PROMPT = "Question: {question}?\n"
FACT_CHECKING_PROMPT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"
ADDITIONAL_BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\nAdditional Background: {additional_background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa": QA_PROMPT,

    'fact_checking': FACT_CHECKING_PROMPT,
}


@torch.no_grad()
def llm_for_open_generation(
        llm, llm_tokenizer,
        prompts,
        retrieval_embeds,
        batch_size=4,
        enable_progress_bar=True,
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx + batch_size]
                       for idx in range(0, len(prompts), batch_size)]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx + batch_size]
                                    for idx in range(0, len(retrieval_embeds), batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)

    progress_bar = tqdm(range(total_test_number), ncols=60,
                        disable=not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_prompt = llm_tokenizer(
            prompt, padding='longest', return_tensors='pt')
        input_ids = tokenized_prompt.input_ids.to(device)
        attention_mask = tokenized_prompt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(
            llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]

            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)

            retrieval_kwargs['retrieval_embeds'] = embeds
            stopping_criteria = stop_sequences_criteria(
                llm_tokenizer, 0, input_ids.shape[0])

        # actual computation
        generated_output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        # because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(
            generated_output[:, input_length:], skip_special_tokens=False)

        results = [x.replace("<unk>", " ") for x in results]

        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers


def format_one_example(
        sample, include_answer, use_rag, use_compression_token, retrieval_embed_length, task_type,
):
    question = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample['background']  # a list
        if isinstance(backgrounds, list):
            backgrounds = " ".join(backgrounds)

        assert isinstance(backgrounds, str), f"backgrounds must be a string, but got: {
        backgrounds}"

        # Use compression tokens instead of COMPRESS
        if use_compression_token:

            if backgrounds.strip() == '':
                backgrounds = []
                compression_tokens = ""

            else:
                backgrounds = sent_tokenize(backgrounds)
                compression_tokens_li = [COMPRESS] * len(backgrounds)
                compression_tokens = "".join(compression_tokens_li)

                # Use compression tokens as background
                # prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(
                #     dict(background=compression_tokens)) + prompt

                # Use compression tokens as additional background
                prompt = ADDITIONAL_BACKGROUND_PROMPT_TEMPLATE.format_map(
                    dict(background=" ".join(backgrounds), additional_background=compression_tokens)) + prompt

        else:

            # Use natural language text as background
            prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(
                dict(background=backgrounds)) + prompt

    return prompt, backgrounds


def get_n_shot_prompt(dev_data, n_shot, task_type, use_rag=False, retrieval_embed_length=0):
    assert n_shot >= 0, n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt, background = format_one_example(
                example, include_answer=True, use_rag=use_rag, retrieval_embed_length=retrieval_embed_length,
                task_type=task_type)
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt, n_shot_background


def prepare_prompts(
        test_data, task_type, tokenizer,
        n_shot=0, use_rag=False,
        use_compression_token=False,
        retrieval_embed_length=0,
        chat_format=None,
):
    if use_compression_token:
        assert use_rag
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx, sample in enumerate(tqdm(test_data, desc="Preparing test data")):
        n_shot = original_n_shot

        while True:
            prompt_start = get_start_prompt(
                task_type, use_rag=use_rag, sample=sample)
            prompt_end, background = format_one_example(
                sample, include_answer=False, use_rag=use_rag, use_compression_token=use_compression_token,
                retrieval_embed_length=retrieval_embed_length, task_type=task_type)

            """
            if 'subject' not in sample.keys():
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d['subject'] == sample['subject']:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects)==5,sample['subject']
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data_with_same_subjects,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            
            if n_shot_prompt:  
                prompt = prompt_start + splitter + splitter.join(n_shot_prompt) + splitter + prompt_end  
            else: 
                prompt = prompt_start + splitter + prompt_end

            """

            prompt = prompt_start + splitter + prompt_end

            if use_compression_token:
                assert prompt.count(COMPRESS) == len(
                    background), (prompt.count(COMPRESS), len(background))

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                if chat_format == "mistral":
                    prompt = create_prompt_with_mistral_chat_format(
                        messages, tokenizer)

                else:
                    raise ValueError(f"Invalid chat format: {chat_format}")

            suffix = reformat_prompt_qasper(question, question_type)
            prompt += suffix

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break

        prompts.append(prompt)
        backgrounds.append(background)

    return prompts, backgrounds


def load_dataset_for_rag(data, use_rag, args):
    dev_data = None
    test_path = f"data/eval/{data}/test.jsonl"
    test_data = None
    if os.path.isfile(test_path):
        test_data = read_jsonl(test_path)

    if use_rag:

        test_retrieval_path = os.path.join(
            f"data/eval/{data}/retrieval/{args.retrieval_prefix}", "test.jsonl")
        test_retrieval = read_jsonl(test_retrieval_path)
        assert len(test_retrieval) == len(test_data)
        for idx in range(len(test_data)):
            test_data[idx]['background'] = [test_retrieval[idx]
                                            ['topk'][rank]['text'] for rank in args.retrieval_topk]

        if args.tf_idf_topk > 0:
            assert args.use_rag
            documents = [x['background'][0] for x in test_data]
            keywords = keyword_extraction_with_tfidf(
                documents, topk=args.tf_idf_topk)
            for idx in range(len(test_data)):
                test_data[idx]['background'] = [keywords[idx]]

        if args.compressor_name_or_path is not None and args.compressor_name_or_path.lower() == "intfloat/e5-large-v2":
            for idx in range(len(test_data)):
                test_data[idx]['background'] = ["passage: " +
                                                x for x in test_data[idx]['background']]

    return dev_data, test_data


if __name__ == "__main__":

    args = parse_args("eval")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='left',
        add_eos_token=False,  # import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # load retriever and retriever_tokenizer
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    retrieval_embed_length = 0
    retriever, retriever_tokenizer = None, None
    if args.compressor_name_or_path is not None:
        if args.compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
            print("Loading SFR model...")
            retriever = SFR.from_pretrained(
                args.compressor_name_or_path, torch_dtype=torch.bfloat16)
            retriever_tokenizer = AutoTokenizer.from_pretrained(
                args.compressor_name_or_path)
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()
        retriever = retriever.to(device)

    """
    ## prepare prompt
    dev_data, test_data = load_dataset_for_rag(
        args.data,
        args.use_rag,
        args,
    )
    """

    test_data = []

    for file_name in args.eval_file:
        test_data += read_jsonl(file_name)

    if args.max_test_samples is not None:
        test_data = test_data[:args.max_test_samples]
        print(f"Using {len(test_data)} samples for evaluation.")

    prompts, backgrounds = prepare_prompts(
        test_data=test_data,
        task_type=args.task_type,
        tokenizer=tokenizer,
        n_shot=args.n_shot,
        use_rag=args.use_rag,
        use_compression_token=args.use_rag and args.compressor_name_or_path is not None,
        retrieval_embed_length=retrieval_embed_length,
        chat_format=args.chat_format,
    )

    retrieval_embeds = None
    if retriever is not None:
        # backgrounds List[List[String]]
        num_samples = len(backgrounds)
        original_orders = []
        # for idx,background in enumerate(backgrounds):
        #     original_orders.extend(
        #         [idx] * len(background)
        #     )
        backgrounds_new = []

        index_start = 0
        for index_background, bg in enumerate(tqdm(backgrounds, desc="Encoding background")):
            assert isinstance(
                bg, list), f"backgrounds must be a list, but got: {bg}"

            backgrounds_new += bg

            # If bg is a list of paragraphs
            # for paragraph in bg:
            #     sents = sent_tokenize(paragraph)
            #     backgrounds_new += sents

            index_end = index_start + len(bg)

            original_orders += [index_background] * len(bg)

            assert len(backgrounds_new) == len(original_orders)

            index_start = index_end

        backgrounds = [x for y in backgrounds for x in y]
        assert len(backgrounds) == len(original_orders) == index_start

        print(f"Preparing document embedding with {
        args.compressor_name_or_path}...")
        _retrieval_embeds = prepare_retrieval_embeds(
            backgrounds,
            retriever,
            retriever_tokenizer,
        )

        retrieval_embeds = [[] for _ in range(num_samples)]
        assert len(_retrieval_embeds) == len(original_orders)
        for id, embeds in zip(original_orders, _retrieval_embeds):
            retrieval_embeds[id].append(embeds)

        for id in range(len(retrieval_embeds)):

            if len(retrieval_embeds[id]) >= 1:
                retrieval_embeds[id] = torch.stack(retrieval_embeds[id])
            else:
                retrieval_embeds[id] = torch.tensor([])

        retriever = retriever.to("cpu")

    avg_prompt_length = tokenizer(prompts, return_length=True).length
    avg_prompt_length = sum(avg_prompt_length) / len(avg_prompt_length)

    # load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = AutoModelForCausalLM
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )

    model.eval()
    # model = model.to(device)
    if retriever is not None:
        assert COMPRESS in tokenizer.get_vocab()
        model.compress_token_id = tokenizer.convert_tokens_to_ids(COMPRESS)

    if args.task_type in ['open_qa', 'fact_checking']:
        generated_results = llm_for_open_generation(
            llm=model,
            llm_tokenizer=tokenizer,
            prompts=prompts,
            retrieval_embeds=retrieval_embeds,
            batch_size=args.eval_batch_size,
            enable_progress_bar=args.enable_progress_bar,
        )
    assert len(generated_results) == len(test_data), len(generated_results)

    output_file_name = os.path.join(args.output_dir, args.model_name_or_path.split('/')[-1], f"{args.output_file}")

    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    generated_results_new = []
    for i in range(len(generated_results)):
        example = {
            'question': test_data[i]['question'],
            'prompt': prompts[i],
            'predict': generated_results[i],
            'answer': test_data[i]['answer'],
            'answerable': test_data[i]['answerable'],
            "question_type": test_data[i].get("question_type", None),
        }
        generated_results_new.append(example)

    write_jsonl(generated_results_new, output_file_name)

    answerables = [entry['answerable'] for entry in test_data]
    generated_results_answerable = [generated for answerable, generated in zip(
        answerables, generated_results) if answerable]
    answers_answerable = [entry for answerable,
    entry in zip(answerables, test_data) if answerable]

    assert len(generated_results_answerable) == len(answers_answerable)

    answers_answerable = [x['answer'] for x in answers_answerable]
    if 'substring_match' in args.eval_metrics:
        score, score_per_sample = get_substring_match_score(
            generated_results, answers_answerable)
        score, score_per_sample = get_substring_match_score(
            generated_results_answerable, answers_answerable)

    if 'fact_checking_acc' in args.eval_metrics:
        score, score_per_sample = eval_fact_checking(
            generated_results, answers)

    if 'truthfulqa_f1_rl' in args.eval_metrics:
        f1, rl, f1_scores, rl_scores = eval_truthfulqa(
            generated_results, answers)
        score = f"{f1}-{rl}"
        score_per_sample = [(f1_score, rl_score)
                            for f1_score, rl_score in zip(f1_scores, rl_scores)]

    result_dict = {
        "batch_size": args.eval_batch_size,
        "include_retrieval": args.use_rag,
        "avg_prompt_length": avg_prompt_length,
        "model": args.model_name_or_path,
        f"{args.eval_metrics}": score,
    }

    if args.compressor_name_or_path is not None:
        result_dict['retriever'] = args.compressor_name_or_path
    print(json.dumps(result_dict, indent=4))

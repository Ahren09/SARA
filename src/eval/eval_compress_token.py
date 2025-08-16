"""
Evaluate the effectiveness of the model decoding the compressed token back to natural language.


Created: 2024.10.9


"""

import os
import re
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = '*.pth'  ## not upload ckpt to wandb cloud

## third-party
import logging
import json
from transformers import AddedToken
from functools import partial
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd

from datasets import DatasetDict
from nltk.tokenize import sent_tokenize


from src.utils.eval_utils import load_eval_model
from src.utils.data_utils import load_dataset_for_eval
from src.utils.utility import print_colored
from src.utils.text_utils import extract_entities_and_numerics

from src.eval.eval_collator import collator
## own
from src.model import (
    SentenceBERTEmbedding,
    SFR,
)

from src.utils.eval_utils import (
    stop_sequences_criteria,
)

from src.data.preprocessing import (
    encode_with_chat_format_pretrain, encode_with_chat_format_paragraph,
    encode_with_chat_format_pretrain_instruction
)

from src.const import COMPRESS

from src.utils.eval_utils import get_retrieval_embeds

from src.arguments import parse_args

logger = get_logger(__name__)


# Define a function to process each row of text
def process_row(row):
    texts = sent_tokenize(row["text"])
    texts = [text.strip() for text in texts]
    return texts, [int(row["id"])] * len(texts)


@torch.no_grad()
def main():
    args = parse_args("eval_compress_token")

    assert (
                   "recover_sentences" in args.output_file and "num-sent" in args.output_file) or "recover_paragraph" in args.output_file
    assert args.output_file.endswith(".xlsx")

    set_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    data, original_dataset = load_dataset_for_eval(args)

    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None)
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=args,
        init_kwargs={
            "wandb": {
                "dir": args.workdir,
                "name": args.wandb_exp_name if args.wandb_exp_name is not None else None,
                "notes": args.wandb_exp_note if args.wandb_exp_note is not None else None,
                "save_code": True,
            },
        }
    )
    accelerator.print(json.dumps(vars(args), indent=4))

    """
    data = load_dataset("din0s/asqa", split="dev") 
    data = data.select(range(args.max_eval_samples))
    """

    TYPE = "pretrain"

    reformatted_data = []

    dataset, original_dataset = load_dataset_for_eval(args)
    
    if len(dataset) > args.max_eval_samples:
        dataset = dataset.select(range(args.max_eval_samples))

    if TYPE == "pretrain":

        for row in tqdm(dataset, desc="Reformatting data"):
            id = row['id']
            if isinstance(row['context'], str):
                context = row['context']
            elif isinstance(row['context'], list):
                context = " ".join(row['context'])
            else:
                raise ValueError(f"Context is neither a list nor a string: {row['context']}")

            sentences = sent_tokenize(context)
            for i in range(0, len(sentences), args.num_compressed_sentences):

                reformatted_data.append({"id": id, "text": " ".join(sentences[i: (i + args.num_compressed_sentences)])})

                if isinstance(args.max_eval_samples, int) and len(reformatted_data) >= args.max_eval_samples:
                    break

        test_data = datasets.Dataset.from_list(reformatted_data)


    elif TYPE == "paragraph":

        for row in tqdm(dataset, desc="Reformatting data"):
            id = row['id']
            for context in row['context']:
                reformatted_data.append({"id": id, "text": context})

        test_data = datasets.Dataset.from_list(reformatted_data)

    SPLIT = 'dev'

    raw_datasets = DatasetDict({
        SPLIT: test_data
    })

    if args.exclude_dataset_type is not None:
        for d_type in args.exclude_dataset_type:
            raw_datasets[SPLIT] = raw_datasets[SPLIT].filter(lambda example: example['task_type'] != d_type)

    if "MistralSmall" in args.model_name_or_path:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            args.model_name_or_path,
            # use_fast=args.use_fast_tokenizer,
            token=os.environ['HF_TOKEN']
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            # use_fast=args.use_fast_tokenizer,
            token=os.environ['HF_TOKEN']
        )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     use_fast=args.use_fast_tokenizer,
    # )

    MODEL_CLASS, CONFIG_CLASS = AutoModelForCausalLM, AutoConfig
    if 'mixtral' in args.model_name_or_path.lower() or 'mistral' in args.model_name_or_path.lower():
        tokenizer.padding_side = 'left'
    elif 'llama' in args.model_name_or_path.lower():
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(["<|finetune_right_pad_id|>"])[0]

    retriever = None
    retriever_hidden_size = 4096
    retrieval_embed_length = 1  ## deprecated since ColBERT is not concluded
    retriever_tokenizer = None
    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path, retriever_hidden_size=retriever_hidden_size)
    config.retriever_hidden_size = retriever_hidden_size
    config.dtype = torch.bfloat16

    if not args.debug:
        # model_original = AutoModelForCausalLM.from_pretrained(args.base_model)
        t0 = time.time()
        # with torch.profiler.profile() as prof:
        #     with torch.profiler.record_function("init model"):

        if args.compressor_name_or_path is not None:
            if args.compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
                retriever = SFR.from_pretrained(args.compressor_name_or_path, torch_dtype=torch.bfloat16)
                retriever.to(args.device)
                retriever_tokenizer = AutoTokenizer.from_pretrained(args.compressor_name_or_path)

            else:
                retriever = SentenceBERTEmbedding(
                    args.compressor_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    # trust_remote_code=True,
                    # token=os.environ['HF_TOKEN']
                )
                retriever_tokenizer = AutoTokenizer.from_pretrained(args.compressor_name_or_path,
                                                                    token=os.environ['HF_TOKEN'])
            retrieval_embed_length = retriever.get_embed_length()
            retriever_hidden_size = retriever.get_embed_dim()
            retriever.eval()

        if retriever is not None:
            retriever = retriever.to(args.device)

        """
        if False: #"llama" in args.model_name_or_path.lower():
        
            model_ori = AutoModelForCausalLM.from_pretrained(
                args.base_model, # the base checkpoint. We need this because we only train the MM projector
                # config=config,
                low_cpu_mem_usage=True,
                device_map='cpu',
                torch_dtype=torch.bfloat16, # torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
                ignore_mismatched_sizes=True
            )
            
            model = MODEL_CLASS(config)
            
            state_dict_ori = model_ori.state_dict()
            state_dict_new = model.state_dict()
            updated_state_dict = {}
            
            saved_params = torch.load(os.path.join(args.model_name_or_path, "ckpt.pth"))
            saved_params = {k.replace('base_model.model.', ''): v.to(torch.bfloat16) for k, v in saved_params.items()}

            for key, param in state_dict_ori.items():
                # print(key)
                if key in state_dict_new:
                    if state_dict_new[key].shape == param.shape:
                        updated_state_dict[key] = param  # Copy matching parameters
                    else:
                        assert key in saved_params, f"Key {key} not found in saved parameters."
                        updated_state_dict[key] = saved_params[key]  # Use saved parameters for mismatched shapes
                    
                else:
                    # This should not happen
                    raise Exception(f"New state dict does not contain key {key}")
                
            # Load saved parameters with strict=False
            missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)
            
            assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
            assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
            
            del model_ori, state_dict_ori, state_dict_new, updated_state_dict
            torch.cuda.empty_cache()
            
        else:
            model = MODEL_CLASS.from_pretrained(
                args.model_name_or_path,
                # config=config,
                low_cpu_mem_usage=True,
                device_map='cpu',
                torch_dtype=torch.bfloat16, # torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
                # ignore_mismatched_sizes=True
            )
        """

        model, tokenizer = load_eval_model(args.model_name_or_path, args.quantization, device=args.device)

        print(f"Loading model:\t{time.time() - t0:.2f} seconds")
        model.retriever_hidden_size = config.retriever_hidden_size
        # prof.export_chrome_trace("profile_trace_.json")
        vocab_size = len(tokenizer)

        if model.get_input_embeddings().weight.shape[0] != vocab_size:
            model.resize_token_embeddings(vocab_size)

        print(f"Loading base model:\t{time.time() - t0:.2f} seconds")

        num_added_tokens = 0
        ## mistral tokenizer is also a LLamaTokenizer
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [0,
                                        1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

        ## COMPRESS simply functions as a placeholder, would not be trained
        num_added_tokens += tokenizer.add_tokens([AddedToken(COMPRESS, lstrip=False, rstrip=False)])
        compress_token_id = tokenizer.convert_tokens_to_ids(COMPRESS)
        model.compress_token_id = compress_token_id
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
        vocab_size = len(tokenizer)
        
       

    if TYPE == 'pretrain':
        if False:  # "llama" in args.model_name_or_path.lower() or "-chat" in args.model_name_or_path.lower():
            encode_function = partial(
                encode_with_chat_format_pretrain_instruction,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                retrieval_embed_length=retrieval_embed_length,
                chat_format=args.chat_format,
                model_name_or_path=args.model_name_or_path
            )

        else:

            encode_function = partial(
                encode_with_chat_format_pretrain,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                retrieval_embed_length=retrieval_embed_length,
                chat_format=args.chat_format,
                do_train=False,
                model_name_or_path=args.model_name_or_path
            )

    elif TYPE == 'paragraph':
        encode_function = partial(
            encode_with_chat_format_paragraph,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            chat_format=args.chat_format,
            do_train=False,
            model_name_or_path=args.model_name_or_path
        )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.num_workers,
        load_from_cache_file=False,
        remove_columns=[name for name in raw_datasets[SPLIT].column_names if
                        name not in ["input_ids", "labels", "attention_mask"]],
        desc=f"Tokenizing and reformatting data",
    )
    lm_datasets.set_format(type="pt")
    if args.task_type == 'finetune':
        lm_datasets[SPLIT] = lm_datasets[SPLIT].filter(lambda example: (example['labels'] != -100).any())
        if args.alpha_kl is not None and args.alpha_kl > 0.0:
            lm_datasets[SPLIT] = lm_datasets[SPLIT].filter(
                lambda example:
                (example['labels'] != -100).sum() == (example['compressor_labels'] != -100).sum()
            )

    dev_dataset = lm_datasets[SPLIT]

    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
    )

    # DataLoaders creation:

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size
    )
    # Freeze model
    for n, p in model.named_parameters():
        p.requires_grad = False

    model.to(args.device)
    model.to(torch.bfloat16)
    logger.info("***** Running eval *****")
    logger.info(f"  Num examples = {len(dev_dataset)}")

    results = {"predictions": [],
               "references": [],
               "num_tokens": [],
               "confidence": []
               }

    model.eval()
    for batch_index, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):

        if batch_index >= 50:
            break

        ## forward with retrieval embeds
        retrieval_kwargs = {}
        if retriever is not None:
            if args.compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':

                retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds(
                    model=retriever,
                    input_ids=batch['compressor_input_ids'].to(args.device),
                    attention_mask=batch['compressor_attention_mask'].to(args.device),
                )

            else:
                text = retriever_tokenizer.batch_decode(batch['compressor_input_ids'], skip_special_tokens=True)
                retrieval_kwargs['retrieval_embeds'] = retriever.get_embedding(
                    text, batch_size=args.per_device_eval_batch_size
                ).to(model.device)

            # Check the decoded text by: retriever_tokenizer.decode(batch['compressor_input_ids'][0])

        input_ids = batch['input_ids'].to(args.device)

        stopping_criteria = stop_sequences_criteria(tokenizer, input_ids.shape[1], input_ids.shape[0])
        
        torch._dynamo.disable(model)
        torch._dynamo.config.suppress_errors = True
         
        
        # with torch._dynamo.disable():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=batch['attention_mask'].to(args.device),
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            **retrieval_kwargs,
        )

        # print(outputs)
        # print(tokenizer.convert_ids_to_tokens(outputs[0]))

        confidences = []

        logits_list = outputs.scores

        BATCH = True
        if BATCH:

            # Batched version
            predicted_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            ground_truths = raw_datasets['dev']['text'][batch_index * args.per_device_eval_batch_size:
                                                        (batch_index + 1) * args.per_device_eval_batch_size]

            # Evaluate Model Confidence

            attention_masks = batch['compressor_attention_mask'].to(args.device)
            # Get the predicted token IDs for all sequences
            predicted_tokens = outputs.sequences

            # Mask to exclude padding tokens
            pad_mask = (predicted_tokens != tokenizer.pad_token_id)  # Shape: (batch_size, seq_len)

            non_pad_counts = torch.sum(pad_mask, dim=1)  # Count non-padding tokens per sequence

            """
            
            # Stack logits for all generation steps into a single tensor (sequence_length, batch_size, vocab_size)
            stacked_logits = torch.stack(logits_list, dim=0)  # Shape: (seq_len, batch_size, vocab_size)

            

            # Compute probabilities for all logits in one go
            probabilities = F.softmax(stacked_logits, dim=-1)  # Shape: (seq_len, batch_size, vocab_size)

            

            predicted_tokens = predicted_tokens.T  # Shape: (seq_len, batch_size)


            # Gather probabilities for the predicted tokens
            predicted_token_probs = torch.gather(
                probabilities, 2, predicted_tokens.unsqueeze(-1)
            ).squeeze(-1)  # Shape: (seq_len, batch_size)

            # Transpose to match (batch_size, seq_len) for masking
            predicted_token_probs = predicted_token_probs.permute(1, 0)  # Shape: (batch_size, seq_len)

            # Apply the mask to exclude padding tokens
            predicted_token_probs = predicted_token_probs * pad_mask.float()  # Shape: (batch_size, seq_len)

            # Compute logit sums and average confidences in a vectorized manner
            logits = torch.log(predicted_token_probs + 1e-12)

            logit_sums = torch.sum(logits * pad_mask, dim=1)  # Sum across sequence length
            
            
            # Average confidence per sequence
            avg_confidences_batch = torch.exp(logit_sums / (pad_mask.sum(dim=1) + 1e-12))
            
            """

            # Store predictions and references
            # results["confidence"].extend(avg_confidences_batch.tolist())
            results["predictions"].extend(predicted_texts)
            results["references"].extend(ground_truths)

            for predicted_text, ground_truth in zip(predicted_texts, ground_truths):
                print("-" * 20)
                print_colored(f"Ground-truth: {ground_truth}", "blue")
                print_colored(f"Model: {predicted_text}", "yellow")
            results['num_tokens'].extend(non_pad_counts.tolist())



        else:

            # unbatched_version
            for i in range(input_ids.shape[0]):
                ground_truth = raw_datasets['dev']['text'][i + batch_index * args.per_device_eval_batch_size]
                predicted_text = tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                predicted_tokens = outputs.sequences[i].tolist()

                # List of logits for each generation step
                attention_mask = batch['compressor_attention_mask'][i].to(
                    args.device)  # Attention mask for this sequence

                probabilities = F.softmax(outputs.scores[i], dim=-1)
                # token_probs = probabilities[range(probabilities.size(0)), predicted_text]
                # confidences.append(token_probs)

                results["predictions"].append(predicted_text)
                results["references"].append(ground_truth)

                print_colored(f"Ground-truth: {ground_truth}", "blue")
                print_colored(f"Model: {predicted_text}", "yellow")

                token_probs = []

                for t, logits in enumerate(logits_list):
                    # Apply softmax to get probabilities
                    probabilities = F.softmax(logits, dim=-1)

                    # Get token probability for the generated token at this step
                    token_id = predicted_tokens[t]  # Token predicted at time step t
                    if token_id == tokenizer.pad_token_id:  # Skip padding tokens
                        continue

                    token_prob = probabilities[i, token_id]  # Select probability for the token
                    token_probs.append(token_prob)

                # Compute logit sum or average confidence
                if token_probs:
                    logit_sum = sum(torch.log(torch.tensor(token_probs)))
                    avg_confidence = torch.exp(logit_sum / len(token_probs))
                else:
                    logit_sum = 0.0
                    avg_confidence = 0.0

                confidences.append(avg_confidence.item())

    if isinstance(results["references"][0], list):
        results["references"] = [" ".join(sents) for sents in results["references"]]

    if "confidence" in results and len(results['confidence']) == 0:
        del results["confidence"]

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_df.to_excel(args.output_file, index=False)


if __name__ == "__main__":
    main()

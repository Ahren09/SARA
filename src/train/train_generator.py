## built-in
import os

# Must run BEFORE importing transformers/datasets/wandb so protobuf uses its
# pure-Python parser, sidestepping the "Descriptors cannot be created directly"
# error from stale _pb2.py files in transitive deps without downgrading protobuf.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_IGNORE_GLOBS", "*.pth")  ## not upload ckpt to wandb cloud

import json
import logging
import math
import sys
import pickle
from warnings import warn
from transformers import Trainer

## third-party
import datasets
import torch
import torch.distributed as dist
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
import torch
from transformers import BitsAndBytesConfig, TrainingArguments
import wandb


import transformers
from argparse import Namespace
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    get_scheduler,
    AutoModelForCausalLM,
    AutoConfig,
    
)
from tokenizers import AddedToken
from src.model.xMistral import XMistralConfig, XMistralForCausalLM
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', "model")))

## own
from ..model import (
    SFR, SentenceBERTEmbedding,
)
from ..utils.data_utils import (
    keyword_extraction_with_tfidf,
    read_jsonl
)

from ..arguments import parse_args

from ..utils.eval_utils import (
    get_retrieval_embeds,
    get_retrieval_embeds_sentence_transformer
)

from ..utils.config_utils import (
    update_config
)

from ..utils.lm_utils import (
    get_nll_loss,
    get_kl_loss,
    save_with_accelerate
)

from ..const import COMPRESS
from ..prompts import MISTRAL_CHAT_TEMPLATE

from ..data.preprocessing import (
    encode_with_chat_format_pretrain,
    encode_with_chat_format_pretrain_instruction,
    encode_with_chat_format_finetune,
    encode_with_chat_format_finetune_stage1,
    encode_with_chat_format_paragraph
)
from ..eval.eval_collator import collator



logger = get_logger(__name__)



@dataclass
class quantization_config:
    quant_type: str =  "fp4" # "fp4" or "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self, quantization: str) -> BitsAndBytesConfig:
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization must be either '4bit' or '8bit'")

        if quantization == "4bit":
            config_params = {
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_quant_storage": self.quant_storage,
            }
            
            return BitsAndBytesConfig(load_in_4bit=True, **config_params)
        else:
            return BitsAndBytesConfig(load_in_8bit=True)


@torch.no_grad()
def validate_during_pretrain(model, dataloader, accelerator, vocab_size, retriever):
    model.eval()
    total_loss = []
    
    
    
    for batch in tqdm(dataloader, desc="Validating"):
        retrieval_embeds = get_retrieval_embeds(
            model=retriever,
            input_ids=batch['compressor_input_ids'],
            attention_mask=batch['compressor_attention_mask'],
        )
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            retrieval_embeds=retrieval_embeds,
        )
        
        
        
        nll_loss = get_nll_loss(
            labels=batch['labels'],
            logits=outputs.logits,
            vocab_size=vocab_size,
        )
        total_loss.append(nll_loss.item())
    model.train()
    if accelerator.use_distributed and accelerator.num_processes > 1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects, total_loss)
        total_loss = [x for y in all_ranks_objects for x in y]
    ppl = torch.exp(torch.tensor(sum(total_loss) / len(total_loss)))
    avg_loss = sum(total_loss) / len(total_loss)
    return ppl, avg_loss


# Define a function to process each row of text
def process_row(row):
    texts = sent_tokenize(row["text"])
    texts = [text.strip() for text in texts]
    return texts, [int(row["id"])] * len(texts)


# Parallelize processing
def parallelize_process_pretrain_dataset(raw_datasets, split: str, num_workers: int = 32):
    new_dataset = {"text": [], "index": []}
    with Pool(min(num_workers, 1)) as pool:
        # Map the processing function to each row in the dataset
        results = list(
            tqdm(pool.imap(process_row, raw_datasets[split]),  # Dataset.from_dict(raw_datasets[split][:100])),
                 total=len(raw_datasets[split]),
                 desc=f"Encoding {split}")
        )

    # Flatten the list of results and add to the new dataset
    for texts, index in results:
        new_dataset["text"].extend(texts)
        new_dataset["index"].extend(index)

    return new_dataset


class CustomTrainer(Trainer):
    def __init__(self, *args, retriever=None, vocab_size=None, additional_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever
        self.vocab_size = vocab_size
        self.additional_args = additional_args
        
    def get_train_dataloader(self):
        # Don't remove unused columns
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(**inputs)
        # print(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        # Compute custom losses as per your original training loop
        loss, logs = None, {}
        if self.additional_args.alpha_nll is not None and self.additional_args.alpha_nll > 0.0:
            nll_loss = get_nll_loss(
                            labels=inputs['labels'],
                            logits=outputs.logits,
                            vocab_size=self.vocab_size,
                        )
            logs["nll_loss"] = nll_loss.detach().item()
            loss = self.additional_args.alpha_nll * nll_loss
        if self.additional_args.alpha_kl is not None and self.additional_args.alpha_kl > 0.0:
            with torch.no_grad():
                model.eval()
                teacher_outputs = model(input_ids=inputs['input_ids_nl'], attention_mask=inputs['attention_mask_nl'])
                model.train()
                
            kl_loss = get_kl_loss(
                teacher_logits=teacher_outputs.logits,
                teacher_labels=inputs['labels_nl'],
                student_logits=outputs.logits,
                student_labels=inputs['labels'],
                temperature=self.additional_args.kl_temperature,
                distill_topk=self.additional_args.distill_topk,
            )
            
            logs["kl_loss"] = kl_loss.detach().item()
            
            self.log(logs)
            
            if loss is not None:
                loss += self.additional_args.alpha_kl * kl_loss
            else:
                loss = self.additional_args.alpha_kl * kl_loss
            loss = loss + self.additional_args.alpha_kl * kl_loss if loss is not None else self.additional_args.alpha_kl * kl_loss

        return (loss, outputs) if return_outputs else loss


    # Override other methods as needed to match your training loop



def main(**kwargs):
    """
    Partly adopted from https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_recipes/finetuning.py
    """
    
    # wandb.login(key=os.environ['WANDB_API_KEY'])
    # os.environ["WANDB_PROJECT"] = kwargs['wandb_project']
    # os.environ["WANDB_NAME"] = kwargs['wandb_exp_name']
    # run = wandb.init(project=kwargs['project_name'], job_type="training",
    #                  config=kwargs)
    
    args = parse_args('train')

    assert args.model_name_or_path is not None, f"model_name_or_path is {model_name_or_path}."
    assert args.output_model_name is not None, "Please specify the output model name."
    set_seed(args.seed)
    ## we need to load retriever before accelerator init
    
    if os.path.basename(args.model_name_or_path).startswith("checkpoint-"):
        tokenizer_and_config_path = os.path.dirname(args.model_name_or_path)
            
    else:
        tokenizer_and_config_path = args.model_name_or_path
    
    
    retriever = None
    retriever_hidden_size = -1
    retrieval_embed_length = 0  ## deprecated since ColBERT is not concluded
    retriever_tokenizer = None
    if args.task_type in {"pretrain", "paragraph"} or (args.task_type in {"finetune", "finetune_stage1"} and args.compressor_name_or_path is not None and args.num_additional_evidence > 0):
        if args.compressor_name_or_path is not None:
            if args.compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
                retriever_kwargs = {
                    "torch_dtype": torch.bfloat16,
                }
                if "checkpoint" in args.compressor_name_or_path:
                    retriever_kwargs["local_files_only"] = True
                
                retriever = SFR.from_pretrained(args.compressor_name_or_path, **retriever_kwargs)
                
                tokenizer_kwargs = {}
                if "checkpoint" in args.compressor_name_or_path:
                    tokenizer_kwargs["local_files_only"] = True
                
                retriever_tokenizer = AutoTokenizer.from_pretrained(args.compressor_name_or_path, **tokenizer_kwargs)
            else:
                retriever = SentenceBERTEmbedding(
                    args.compressor_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    # trust_remote_code=True,
                    # token=os.environ['HF_TOKEN']
                )
                retriever_tokenizer = AutoTokenizer.from_pretrained(args.compressor_name_or_path, token=os.environ['HF_TOKEN'])
        retrieval_embed_length = retriever.get_embed_length()  # Seems always 1
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()
    args.wandb_exp_name = f"{args.task_type}_{tokenizer_and_config_path.split('/')[-1]}"
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with="wandb" if args.use_wandb else None)
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
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")

        checkpoint_dir = [os.path.abspath(os.path.join(args.output_dir, "..", 'checkpoints'))]

    if accelerator.use_distributed: dist.broadcast_object_list(checkpoint_dir, src=0)
    args.output_dir = checkpoint_dir[0]

    if retriever is not None:
        retriever = retriever.to(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.dev_file is not None:
        data_files['dev'] = args.dev_file

    """
    # Ahren: Split sentences in TriviaQA then encode using SARA compressor

    data = load_dataset("din0s/asqa", split="train")

    long_answers = {}

    for row_index, row in enumerate(tqdm(data, desc=f"{args.dataset_name}")):
        if row_index not in long_answers:
            long_answers[row_index] = []

        long_answers[row_index] = []
        sentences = sent_tokenize(row['annotations'][0]['long_answer'])

        long_answers[row_index] = sentences

    # dev_data, test_data = load_jsonl_dataset(
    #     args.dataset_name,
    #     use_rag=True,
    #     args=args,
    # )

    """
    
    
    def standardize_context(example):
        """
        When the feature is string, convert it into a list with a single string.
        """
        return {"context": example["context"] if isinstance(example["context"], list) else [example["context"]]}
    
    
    def standardize_additional_context(example):
        example["additional_context"] = [
            str(item) for item in example["additional_context"]
        ] if example["additional_context"] and len(example["additional_context"]) > 0 else [""]
        return example



    if args.task_type in {'pretrain', 'paragraph'}:

        if args.debug and args.max_train_samples is not None:
            import itertools
            OVERSAMPLE = 10 if args.task_type == 'paragraph' else 4
            n_train = (args.max_train_samples + args.index_start) * OVERSAMPLE
            n_dev = args.max_eval_samples * OVERSAMPLE if args.max_eval_samples < 10**8 else 1000

            streamed = load_dataset(
                "json",
                data_files=data_files,
                streaming=True,
                **dataset_args,
            )
            train_rows = list(itertools.islice(streamed["train"], n_train))
            raw_datasets = datasets.DatasetDict({"train": Dataset.from_list(train_rows)})
            if args.dev_file is not None and "dev" in streamed:
                dev_rows = list(itertools.islice(streamed["dev"], n_dev))
                raw_datasets["dev"] = Dataset.from_list(dev_rows)
        else:
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )

            raw_datasets = datasets.DatasetDict({
                "train": raw_datasets["train"].select(range(args.max_train_samples)) if args.debug else raw_datasets["train"],
                "dev": raw_datasets["dev"].select(range(args.max_eval_samples)) if args.debug else raw_datasets["dev"],
            })

        # We don't worry about evidence being too long
        # train_dataset = parallelize_process_pretrain_dataset(raw_datasets, 'train', args.num_workers)
        # dev_dataset = parallelize_process_pretrain_dataset(raw_datasets, 'dev', args.num_workers)
        
        def filter_by_length(example, min_chars=32, max_chars=32768):
            text = example["text"]
            if min_chars <= len(text) <= max_chars:
                return True
            return False
        
        
        if args.task_type == 'paragraph':
            min_len, max_len = 128, 2048
            train_dataset = raw_datasets['train'].filter(lambda x: filter_by_length(x, min_chars=min_len, max_chars=max_len))
            dev_dataset = raw_datasets['dev'].filter(lambda x: filter_by_length(x, min_chars=min_len, max_chars=min_len))
            
        elif args.task_type == 'pretrain':
            min_len = 32
            train_dataset = raw_datasets['train'].filter(lambda x: filter_by_length(x, min_chars=min_len))
            dev_dataset = raw_datasets['dev'].filter(lambda x: filter_by_length(x, min_chars=min_len))
            
        
        # train_dataset['id'] = range(len(train_dataset['text']))
        # dev_dataset['id'] = range(len(dev_dataset['text']))
        
        train_dataset = train_dataset.select(range(args.index_start, args.max_train_samples + args.index_start)) if (args.max_train_samples is not None and len(train_dataset) > args.index_start + args.max_train_samples) else train_dataset
        dev_dataset = dev_dataset.select(range(args.max_eval_samples)) if (args.max_eval_samples is not None and len(dev_dataset) > args.max_eval_samples) else dev_dataset

        if args.debug and args.max_train_samples is not None and len(train_dataset) < args.max_train_samples:
            logger.warning(
                f"After filter_by_length, train_dataset has {len(train_dataset)} rows "
                f"(requested {args.max_train_samples}). Increase the OVERSAMPLE multiplier if needed."
            )

        lengths = [len(t) for t in train_dataset['text'][:1000]]
        print(f"Max length (first 1000 examples): {max(lengths)}, Mean length: {sum(lengths)/len(lengths)}")

        raw_datasets = DatasetDict({
            "train": train_dataset,
            "dev": dev_dataset
        })




    elif args.task_type in {"finetune", "finetune_stage1"}:
        
        if isinstance(args.train_file, str):
            train_file_list = [args.train_file]
            
        elif args.train_file is None:
            train_file_list = []
            
        else:
            assert isinstance(args.train_file, list), "Please specify the train_file as a list of files."
            train_file_list = args.train_file
        
        if hasattr(args, "multiple_choice_train_file") and args.multiple_choice_train_file is not None:
            train_file_list = args.multiple_choice_train_file + train_file_list
            
        
        data_list = []
        
        

        
         #  "question_type", "answerable"]
        for train_file in train_file_list:
        
            logger.info(f"Loading {train_file}")
            if train_file.endswith(".jsonl"):
                logger.info(f"Loading {train_file}")
                data = read_jsonl(
                    train_file,
                    max_lines=args.max_train_samples if (args.debug and args.max_train_samples) else None,
                )
                if args.max_train_samples is not None:
                    data = data[:args.max_train_samples]

                raw_datasets = Dataset.from_list(data)

            elif train_file.endswith(".json"):
                data = json.load(open(train_file))
                if isinstance(data, dict):
                    raw_datasets = Dataset.from_dict(data)

                elif isinstance(data, list):
                    if args.max_train_samples is not None:
                        data = data[:args.max_train_samples]
                    raw_datasets = Dataset.from_list(data)

                else:
                    raise ValueError(f"Unsupported file format for train_file: {args.train_file}")

            else:
                raise ValueError(f"Unsupported file format for train_file: {args.train_file}")
            
            existing_features = list(raw_datasets.features.keys())
            if "multi-choice" in train_file:
                assert "QuALITY" in train_file
                features_to_keep = ["id", "example_id","question", "context", "correct_choice", "choices"]
                raw_datasets = raw_datasets.select_columns(features_to_keep)
                raw_datasets = raw_datasets.rename_column("correct_choice", "answer")
                # raw_datasets = raw_datasets.map(lambda x: {"answer_reformatted": x["answer"]})
                raw_datasets = raw_datasets.map(lambda x: {"question_type": "multi-choice"})
            
            elif "answer_reformatted" in existing_features:
                features_to_keep = ["id", "example_id","question", "context", "answer_reformatted"] #, "answer_reformatted"]
                raw_datasets = raw_datasets.select_columns(features_to_keep)
                raw_datasets = raw_datasets.map(lambda x: {"question_type": "open_qa"})
                raw_datasets = raw_datasets.map(lambda x: {"choices": [""]})
                raw_datasets = raw_datasets.rename_column("answer_reformatted", "answer")
                
            
            
            else:
                # raise ValueError(f"Unsupported file format for train_file: {args.train_file}")
                # This should not happen
                features_to_keep = ["id", "example_id","question", "context", "answer"] 
                raw_datasets = raw_datasets.select_columns(features_to_keep)
                # raw_datasets = raw_datasets.map(lambda x: {"answer_reformatted": x["answer"]})
                raw_datasets = raw_datasets.map(lambda x: {"question_type": "open_qa"})
                raw_datasets = raw_datasets.map(lambda x: {"choices": [""]})
                
            # print(raw_datasets)
            
            def filter_by_num_context(example, min_context: int):
                if len(example["context"]) >= min_context:
                    return True
                return False
            
            if args.num_additional_evidence > 0 or args.task_type in {'finetune', 'finetune_stage1'}:
                raw_datasets = raw_datasets.filter(lambda x: 
                    True if (isinstance(x['answer'], str) or (isinstance(x['answer'], list) and len(x['answer']) > 0)) else False).filter(
                        lambda x: filter_by_num_context(x, min_context=args.num_evidence))
            
            raw_datasets = raw_datasets.map(lambda x: {"answer": x["answer"] if isinstance(x["answer"], list) else [x["answer"]]})
            # raw_datasets = raw_datasets.map(lambda x: {"answer_reformatted": x["answer_reformatted"] if isinstance(x["answer_reformatted"], list) else [x["answer_reformatted"]]})
                
            # raw_datasets = raw_datasets.select(range(100))
            data_list.append(raw_datasets)
            
        
            
            concat_datasets = concatenate_datasets(data_list)
        

        if isinstance(raw_datasets, Dataset):
            raw_datasets = DatasetDict({
                "train": concat_datasets,
            })

        else:
            assert isinstance(concat_datasets, DatasetDict), "raw_datasets should be a Dataset or DatasetDict object."
            raw_datasets = concat_datasets
            # raw_datasets = load_dataset(
            #     "json",
            #     data_files=data_files,
            #     **dataset_args,
            # )

    else:
        raise ValueError(f"task_type {args.task_type} not supported")

    """
    # Is this necessary?
    if args.task_type == 'paragraph':

        for split in ['train', 'dev']:
            raw_datasets[split] = raw_datasets[split].to_pandas().groupby('index').agg(
                {'text': list, 'id': list}).reset_index().sort_values('index')
            raw_datasets[split] = Dataset.from_pandas(raw_datasets[split])
    """
    if args.exclude_dataset_type is not None:
        for d_type in args.exclude_dataset_type:
            raw_datasets['train'] = raw_datasets['train'].filter(lambda example: example['task_type'] != d_type)

    """
    df = pd.DataFrame.from_dict(long_answers, orient='index').stack().reset_index(level=1, drop=True).reset_index()
    df.columns = ['id', 'text']
    df = df.rename({"id": "original_id"}, axis=1)
    df['id'] = df.index


    raw_datasets = Dataset.from_pandas(df)


    # Split the dataset into train and dev (e.g., 80-20 split)
    train_test_split = raw_datasets.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    dev_dataset = train_test_split['test']

    # Construct the DatasetDict
    raw_datasets = DatasetDict({
        "train": train_dataset,
        "dev": dev_dataset
    })
    """

    model_name_or_path = os.path.abspath(
        os.path.join(args.output_dir, "..", args.model_name_or_path)) if args.model_name_or_path.startswith(
        "checkpoint") else args.model_name_or_path
    
    if "mistral" in model_name_or_path.lower() and "small" in model_name_or_path.lower():
        tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_and_config_path,
            use_fast=args.use_fast_tokenizer,
            token=os.environ['HF_TOKEN']
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_and_config_path,
            use_fast=args.use_fast_tokenizer,
            token=os.environ['HF_TOKEN']
        )
        
    # vocab_size = len(tokenizer)
    
    
        
    if tokenizer.chat_template is None and "mistral" in model_name_or_path.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    
    
    
    # if tokenizer.pad_token:
    #     pass
    # elif tokenizer.eos_token:
    #     print("Setting pad_token to eos_token")
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    #     tokenizer.pad_token = tokenizer.eos_token
        
    # elif tokenizer.unk_token:
    #     print("Setting pad_token to unk_token")
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    #     tokenizer.pad_token = tokenizer.unk_token
    
    
    # SARA uses the projector-augmented Mistral. Using Auto* here would silently load a plain Mistral
    # (no projector) and degrade SARA to RAG with no error (release plan §2).
    MODEL_CLASS, CONFIG_CLASS = XMistralForCausalLM, XMistralConfig
    if 'mixtral' in args.model_name_or_path.lower() or 'mistral' in args.model_name_or_path.lower():
        tokenizer.padding_side = 'left'
    elif 'llama' in args.model_name_or_path.lower():
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(["<|finetune_right_pad_id|>"])[0]
    
    bnb_config = None
    if args.quantization:
        # Define 8-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit quantization
            bnb_8bit_compute_dtype=torch.float16,  # Use FP16 for computation to balance speed and precision
            bnb_8bit_use_double_quant=False,  # Disable double quantization (you can enable for extra memory savings)
        )


        if args.quantization == "8bit" and args.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )


    config = CONFIG_CLASS.from_pretrained(tokenizer_and_config_path,
                                          retriever_hidden_size=retriever_hidden_size,
                                          token=os.environ['HF_TOKEN']
                                          )
    
    # config.vocab_size = vocab_size
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Task type (e.g., Causal LM, Seq2Seq LM, Token Classification)
        r=16,                         # Rank of the low-rank adaptation
        lora_alpha=128,                # LoRA scaling factor
        lora_dropout=0.1,             # Dropout for LoRA layers
        target_modules=["q_proj", "v_proj"],  # explicit; matched across RAG and SARA (release plan §3)
        # projector is persisted via save_sara_extras, NOT modules_to_save (PEFT nesting bug, §8)
    )
    
    # if "gemma" in args.model_name_or_path.lower():
    #     model = MODEL_CLASS.from_pretrained(
    #         model_name_or_path,
    #         quantization_config=bnb_config if args.quantization else None,
    #         use_flash_attention_2=args.use_flash_attn,
    #         torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
    #         token=os.environ['HF_TOKEN']
    #     )

    # else:
    model = MODEL_CLASS.from_pretrained(
        model_name_or_path,
        config=config,
        quantization_config=bnb_config if args.quantization else None,
        # transformers>=4.x: use attn_implementation (flash-attn is ABI-broken here -> sdpa).
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
        token=os.environ['HF_TOKEN']
    )
    
    
    num_added_tokens = 0
    ## mistral tokenizer is also a LLamaTokenizer
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer,
                                                           LlamaTokenizerFast) or "mistral" in model_name_or_path.lower():
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0,
                                    1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    num_added_tokens += tokenizer.add_tokens([AddedToken(COMPRESS, lstrip=False, rstrip=False)])
    compress_token_id = tokenizer.convert_tokens_to_ids(COMPRESS)  # 128256 for Llama 3
    model.compress_token_id = compress_token_id
    
    if "gemma" in args.model_name_or_path.lower():
        vocab_size = model.model.embed_tokens.num_embeddings + 1
    else:
        vocab_size = len(tokenizer)

    
    if num_added_tokens > 0:
        # model.resize_token_embeddings(vocab_size)
        model.resize_token_embeddings(vocab_size)
        
    assert model.config.vocab_size == vocab_size
    config.vocab_size = vocab_size

    if args.use_trainer:
        model.enable_input_require_grads()
    
    # Fail loudly if a plain Mistral slipped through (no projector) for a SARA run (release plan §2).
    if not isinstance(model, XMistralForCausalLM):
        raise TypeError(f"Expected XMistralForCausalLM, got {type(model).__name__}.")
    if getattr(args, "num_additional_evidence", 0) and not hasattr(model, "projector"):
        raise RuntimeError("SARA training (num_additional_evidence>0) requires a projector.")

    # Stage-B handoff (release plan §4): warm-start the projector (+ added-token rows) from a prior
    # projector-alignment run before the matched QA fine-tune. LoRA stays freshly initialized.
    if getattr(args, "init_projector_from", None) and hasattr(model, "projector"):
        from src.model.loader import load_sara_extras
        load_sara_extras(model, args.init_projector_from)
        print(f"[SARA] initialized projector from {args.init_projector_from}")

    if args.use_lora:
        print("Setting up PEFT model")
        model = get_peft_model(model, lora_config)
        # The projector trains alongside LoRA (base is otherwise frozen). Persisted via save_sara_extras.
        if hasattr(model.get_base_model(), "projector") and not args.freeze_projector:
            model.get_base_model().projector.requires_grad_(True)
    

    # Preprocessing the datasets.
    if args.task_type == 'finetune_stage1':
        encode_function = partial(
            encode_with_chat_format_finetune_stage1,
            # if "messages" in raw_datasets["train"].column_names else encode_with_completion_format_finetune,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            use_rag_tuning=args.use_rag_tuning,
            retriever_tokenizer=retriever_tokenizer,
            chat_format=args.chat_format,
            num_evidence=args.num_evidence,
            num_additional_evidence=args.num_additional_evidence,
            
            model_name_or_path=args.model_name_or_path
        )
            
    elif args.task_type == 'finetune':
        encode_function = partial(
            encode_with_chat_format_finetune,
            # if "messages" in raw_datasets["train"].column_names else encode_with_completion_format_finetune,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            use_rag_tuning=args.use_rag_tuning,
            use_retriever_embed=args.num_additional_evidence > 0,
            retriever_tokenizer=retriever_tokenizer,
            chat_format=args.chat_format,
            num_evidence=args.num_evidence,
            num_additional_evidence=args.num_additional_evidence,
            model_name_or_path=args.model_name_or_path
        )
            
            
    elif args.task_type == 'pretrain':
        if False: # "llama" in args.model_name_or_path.lower() or "-chat" in args.model_name_or_path.lower():
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
                model_name_or_path=args.model_name_or_path
            )

    elif args.task_type == 'paragraph':
        encode_function = partial(
            encode_with_chat_format_paragraph,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            chat_format=args.chat_format,
            model_name_or_path=args.model_name_or_path
        )

    else:
        raise ValueError(f"task_type {args.task_type} not supported")


    print(f"Tokenizing and reformatting data ...")
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.num_workers,
            load_from_cache_file=False,
            # remove_columns=[name for name in raw_datasets["train"].column_names if
            #                 name not in ["input_ids", "labels", "attention_mask"]],
            desc=f"Tokenizing and reformatting data on rank: {accelerator.local_process_index}",
        )
        column_names = ["input_ids", "labels", "attention_mask", ]
        if args.task_type in {"pretrain", "paragraph"} or (args.task_type in {"finetune", "finetune_stage1"} and args.num_additional_evidence > 0):
            column_names.extend(["retriever_input_text"])
            
        if args.task_type == 'finetune_stage1':
            column_names.extend(["input_ids_nl", "labels_nl", "attention_mask_nl"])
            
        lm_datasets = lm_datasets.select_columns(column_names)
        def filter_long_inputs(example):
            return len(example["input_ids"]) <= args.max_seq_length

        
        lm_datasets.set_format(type="pt")
        if args.task_type == 'finetune':  # TODO
            lm_datasets['train'] = lm_datasets['train'].filter(lambda example: (example['labels'] != -100).any())
            lm_datasets['train'] = lm_datasets['train'].filter(filter_long_inputs)
            


    train_dataset = lm_datasets["train"]
    dev_dataset = lm_datasets['dev'] if args.dev_file is not None else None

    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
        task=args.task_type,
    )

    
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True if args.task_type in {'finetune', "finetune_stage1"} else False,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size
        )
        
    

    if args.update_projector_only:
        for n, p in model.named_parameters():
            if 'projector' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.projector_learning_rate)
    else:
        params_no_decay = []
        params_decay = []
        params_projector = []
        
        for n, p in model.named_parameters():
            
            if 'projector' in n:
                # Set to True and then use `lr` to decide whether we actually train the projector
                p.requires_grad = True
                params_projector.append(p)
            elif 'bias' in n or 'layer_norm.weight' in n:
                params_no_decay.append(p)
            else:
                params_decay.append(p)
                    
            
        if args.freeze_projector:
            
            optimizer_grouped_parameters = [
                {
                    "params": params_no_decay,
                    "weight_decay": 0.0,
                    'lr': args.model_learning_rate
                },
                {
                    "params": params_projector,
                    "weight_decay": 0.0,
                    'lr': args.projector_learning_rate
                },
                {
                    "params": params_decay,
                    "weight_decay": args.weight_decay,
                    'lr': args.model_learning_rate
                },
                
            ]
            
        else:
            
            optimizer_grouped_parameters = [
                {
                    "params": params_no_decay,
                    "weight_decay": 0.0,
                    'lr': args.model_learning_rate
                },
                {
                    "params": params_projector,
                    "weight_decay": 0.0,
                    'lr': args.projector_learning_rate
                },
                {
                    "params": params_decay,
                    "weight_decay": args.weight_decay,
                    'lr': args.model_learning_rate
                },
                
            ]
            
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # # https://github.com/microsoft/DeepSpeed/pull/4966
    # if args.chat_format == 'mixtral':
    #     deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # Prepare everything with `accelerator`.
    if dev_dataset is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, dev_dataloader)

    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    try:
        checkpointing_steps = int(args.checkpointing_steps)
    except:
        checkpointing_steps = None

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max Sequence Length = {args.max_seq_length}")
    logger.info(
        f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / (10 ** 6):.2f} M")  ## not applicable for deepspeed

    completed_steps = 0
    starting_epoch = 0

    # logging_interval_grad_norm = 0
    logging_interval_loss = 0
    logging_interval_kl_loss = 0
    logging_interval_nll_loss = 0

    total_loss = 0
    total_kl_loss = 0
    total_nll_loss = 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar = tqdm(range(args.max_train_steps), disable=True)

    # update the progress_bar if load from checkpoint
    save_one_sample = True
    checkpoint_dir = os.path.join("checkpoints", args.task_type, args.output_model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    json.dump(vars(args), open(os.path.join(checkpoint_dir, "args.json"), 'w'), indent=2)
    
    if args.use_trainer:
        
        model.train()
        
        
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            eval_strategy="no", # "epoch",
            learning_rate=args.model_learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            # max_steps takes precedence over epochs so RAG and SARA get IDENTICAL optimizer updates
            # (release plan §3); effective batch = per_device * grad_accum is also matched.
            max_steps=args.max_train_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            seed=args.seed,
            bf16=True,
            logging_steps=args.logging_steps,
            gradient_checkpointing=True,
            save_total_limit=1,
            report_to=(["wandb"] if args.use_wandb else []),
        )
        
        
                
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            retriever=None,  # Replace with your retriever if available
            vocab_size=len(tokenizer),
            additional_args=args,
        )
        
        trainer.train()
        # trainer.log_metrics("train", train_result.metrics)

        trainer.save_model()
        model.config.save_pretrained(checkpoint_dir)

        # The HF Trainer saves the LoRA adapter (+ resized embeddings) but NOT the projector, which is
        # not a PEFT module. Persist it explicitly so a fresh-process reload is exact (release plan §8).
        if accelerator.is_main_process and args.use_lora:
            from src.model.loader import PAD, save_sara_extras  # COMPRESS already imported at module level
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            if hasattr(base, "projector"):
                added_ids = [tokenizer.convert_tokens_to_ids(PAD),
                             tokenizer.convert_tokens_to_ids(COMPRESS)]
                save_sara_extras(model, added_ids, checkpoint_dir,
                                 int(base.config.retriever_hidden_size))
                print(f"[SARA] saved projector + added-token extras to {checkpoint_dir}")

        
        
    else:
        
        # Decreases the memory usage but increases the training time. See https://huggingface.co/docs/transformers/v4.19.4/en/performance#gradient-checkpointing
        print("Disabling gradient checkpointing")
        model.gradient_checkpointing_enable()


        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            active_dataloader = train_dataloader
            # model.requires_grad_()

            for index_batch, batch in enumerate(active_dataloader):
                """if save_one_sample and index_batch == 0:
                    if accelerator.is_local_main_process:
                        output_dir = os.path.join(args.output_dir, "..", "checkpoints", f"{args.task_type}",
                                                f"{args.model_name_or_path.split('/')[-1]}")
                        os.makedirs(output_dir, exist_ok=True)
                        pickle.dump(
                            batch,
                            open(os.path.join(os.path.dirname(output_dir), "sample_data.pkl"), 'wb'),
                        )
                    print("**" * 20, "show one example", "**" * 20)
                    print(batch.keys())
                    print(tokenizer.decode(batch['input_ids'][0]))
                    print(batch['input_ids'][0])
                    if "retriever_input_text" in batch:
                        print(batch['retriever_input_text'][0])
                    if 'input_ids' in batch:
                        for input_id, label_id, attention_mask in zip(batch['input_ids'][0], batch['labels'][0],
                                                                    batch['attention_mask'][0]):
                            accelerator.print(
                                f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",
                                end=" ")
                    
                    for input_id, label_id, attention_mask in zip(batch['input_ids'][0], batch['labels'][0],
                                                                batch['attention_mask'][0]):
                        accelerator.print(
                            f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",
                            end=" ")
                    print('\n' + "**" * 20, "show one example", "**" * 20)
                    save_one_sample = False
                """
                # with torch.profiler.profile() as prof:
                #     with torch.profiler.record_function("init model"):
                with accelerator.accumulate(model):
                    ## forward with retrieval embeds
                    retrieval_kwargs = {}
                    if retriever is not None:
                        if args.compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
                            
                            
                            retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds(
                                model=retriever,
                                input_ids=batch['compressor_input_ids'],
                                attention_mask=batch['compressor_attention_mask'],
                            ).to(model.device)
                            
                        else:
                            text = retriever_tokenizer.batch_decode(batch['compressor_input_ids'], skip_special_tokens=True)
                            retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds_sentence_transformer(
                                model=retriever,
                                text=text
                            ).to(model.device)

                        # Seems unnecessary
                        # retrieval_kwargs['retriever_example_ids'] = batch['retriever_example_ids']
                    
                    """
                    import pdb; pdb.set_trace()
                    for i in range(batch['input_ids'].shape[0]):
                        print(f"=" * 20)
                        print(f"Batch {index_batch}, {i}")
                        print(f"=" * 20)
                        print(tokenizer.decode(batch['input_ids'][i]))
                        print(tokenizer.decode(batch['labels'][i][batch['labels'][i] != -100]))
                    """
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        **retrieval_kwargs,
                    )
                    loss = None
                    if args.alpha_nll is not None and args.alpha_nll > 0.0:
                        nll_loss = get_nll_loss(
                            labels=batch['labels'],
                            logits=outputs.logits,
                            vocab_size=vocab_size,
                        )

                        logging_interval_nll_loss += nll_loss.detach().float()

                        loss = args.alpha_nll * nll_loss

                    if args.alpha_kl is not None and args.alpha_kl > 0.0:

                        ## forward with retrieval tokens
                        with torch.no_grad():
                            model.eval()
                            teacher_outputs = model(
                                input_ids=batch['input_ids_nl'],
                                attention_mask=batch['attention_mask_nl'],
                            )
                            model.train()

                        kl_loss = get_kl_loss(
                            teacher_logits=teacher_outputs.logits,
                            teacher_labels=batch['labels_nl'],
                            student_logits=outputs.logits,
                            student_labels=batch['labels'],
                            temperature=args.kl_temperature,
                            distill_topk=args.distill_topk,
                        )
                        logging_interval_kl_loss += kl_loss.detach().float()
                        if loss is not None:
                            loss += args.alpha_kl * kl_loss
                        else:
                            loss = args.alpha_kl * kl_loss

                    logging_interval_loss += loss.detach().float()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if args.logging_steps and completed_steps % args.logging_steps == 0:
                        avg_loss = accelerator.gather(
                            logging_interval_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                        total_loss += accelerator.gather(
                            logging_interval_loss).mean().item() / args.gradient_accumulation_steps

                        to_be_logged = {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                            "rolling_loss": total_loss / completed_steps,
                        }
                        if args.alpha_nll is not None and args.alpha_nll > 0.0:
                            total_nll_loss += accelerator.gather(
                                logging_interval_nll_loss).mean().item() / args.gradient_accumulation_steps
                            to_be_logged["rolling_nll_loss"] = total_nll_loss / completed_steps

                        if args.alpha_kl is not None and args.alpha_kl > 0.0:
                            total_kl_loss += accelerator.gather(
                                logging_interval_kl_loss).mean().item() / args.gradient_accumulation_steps
                            to_be_logged["rolling_kl_loss"] = total_kl_loss / completed_steps

                        accelerator.log(to_be_logged, step=completed_steps)

                        # logging_interval_grad_norm = 0
                        logging_interval_loss = 0
                        logging_interval_kl_loss = 0
                        logging_interval_nll_loss = 0

                    if isinstance(checkpointing_steps, int) and (completed_steps + 1) % checkpointing_steps == 0:
                        output_dir = os.path.abspath(os.path.join(args.output_dir, "..", "checkpoints", f"{args.task_type}",
                                    f"{args.output_model_name.split('/')[-1]}", f"step_{completed_steps + 1}"))
                        
                        save_with_accelerate(accelerator, model, tokenizer, output_dir,
                                            save_projector_only=args.update_projector_only, args=args)

                if args.eval_steps is not None and (completed_steps + 1) % args.eval_steps == 0:
                    if dev_dataloader is not None:
                        if args.task_type == 'pretrain':
                            ppl, avg_loss = validate_during_pretrain(model, dev_dataloader, accelerator, vocab_size,
                                                            retriever)
                            
                            
                            
                            accelerator.log({"dev_ppl": ppl, "dev_loss": avg_loss}, step=completed_steps)
                            

                if completed_steps >= args.max_train_steps:
                    break
                # prof.export_chrome_trace(f"outputs/profile_trace_{index_batch}.json")

            if epoch != args.num_train_epochs - 1:  # Skip the last epoch as it will be saved to `final`
                output_dir = os.path.abspath(os.path.join(args.output_dir, "..", "checkpoints", f"{args.task_type}",
                                        f"{args.output_model_name.split('/')[-1]}", f"epoch_{epoch}"))
                save_with_accelerate(accelerator, model, tokenizer, output_dir,
                                    save_projector_only=False, args=args 
                                    )

        accelerator.end_training()
        
    

        ## save the last one
        output_dir = os.path.abspath(os.path.join(args.output_dir, "..", "checkpoints", f"{args.task_type}",
                                f"{args.output_model_name.split('/')[-1]}", f"final"))
        save_with_accelerate(accelerator, model, tokenizer, output_dir, save_projector_only=False, args=args)


if __name__ == "__main__":
    main()

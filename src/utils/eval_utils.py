import unicodedata
from typing import List, Union
import os
import numpy as np
import torch
import transformers
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
from transformers import LlamaTokenizer, LlamaTokenizerFast

from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.prompts import PROMPT_MISTRAL, PROMPT_LLAMA3
from src.model import SentenceBERTEmbedding
from src.model.retriever import getBM25Retriever, getHuggingFaceRetriever
from src.utils.utility import print_colored
from .text_utils import SimpleTokenizer

COMPRESS = "<COMPRESS>"

# Macros

QA_PROMPT = "Question: {question}?\n"
FACT_CHECKING_PROMPT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa": QA_PROMPT,
    'fact_checking': FACT_CHECKING_PROMPT,
}


def add_new_pad_token(tokenizer, model):
    print("Setting <pad> to 32000 for mistral models ...")
    assert tokenizer.pad_token is None
    assert tokenizer.pad_token_id is None
    assert tokenizer.convert_ids_to_tokens([32000])[0] is None
    num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.pad_token_id == 32000


def set_pad_token_to_eos(tokenizer):
    print("Setting <pad> to the same ID as EOS for mistral models ...")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token == "<pad>"


def format_prompt(model_name_or_path, question: str, knowledge: str, additional_context: List[str] = [],
                  answer_instruction: str = None):
    if "llama" in model_name_or_path.lower():
        return format_prompt_llama(question, knowledge, additional_context, answer_instruction)
    elif "mistral" in model_name_or_path.lower():
        return format_prompt_mistral(question, knowledge, additional_context, answer_instruction)


def format_prompt_llama(question: str, knowledge: str, additional_context: List[str] = [],
                        answer_instruction: str = None):
    if answer_instruction is None:
        answer_instruction = "Answer the question based on the given context. Keep your answer short."

    if additional_context:
        background = "## Additional Context"
        for idx, bg in enumerate(additional_context):
            background += f"\n{idx + 1}: {COMPRESS}"


    else:
        additional_context = ""

    prompt = PROMPT_LLAMA3.format(knowledge=knowledge, background=background, question=question,
                                  answer_instruction=answer_instruction)
    return prompt


def format_prompt_mistral(question: str, knowledge: str, additional_context: List[str] = [],
                          answer_instruction: str = None):
    """
    Mistral prompt format:
    <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
    
    Reference: https://www.promptingguide.ai/models/mistral-7b
    
    """

    if answer_instruction is None:
        answer_instruction = "Answer the question based on the given context. Keep your answer short."

    if additional_context:
        background = "## Additional Context"
        for idx, bg in enumerate(additional_context):
            background += f"\n{idx + 1}: {COMPRESS}"


    else:
        additional_context = ""

    prompt = PROMPT_MISTRAL.format(knowledge=knowledge, background=background, question=question,
                                   answer_instruction=answer_instruction)

    return prompt


def load_hf_pipeline(model_name_or_path: str, quantization: str, temperature=0.001, max_new_tokens: int = 64,
                     device: str = "cuda",
                     hf_token=None):
    tokenizer_kwargs = {}
    model_kwargs = {}
    
    if "checkpoint" in model_name_or_path:
        tokenizer_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=min(temperature, 0.001),
                    repetition_penalty=1.1,
                    return_full_text=False,
                    )

    return pipe


def load_eval_model(model_name: str, quantization: str, hf_token=None, device: str="cuda"):
    """Load a trained SARA/RAG model for evaluation (LoRA-first, public-base + adapter).

    If ``model_name`` is a SARA adapter directory (contains ``adapter_config.json``), delegate to the
    single source of truth :func:`src.model.loader.load_sara_for_eval`, which rebuilds the public base
    as :class:`XMistralForCausalLM`, restores the projector + added-token rows, and loads the LoRA
    adapter. This guarantees SARA never silently degrades to a plain Mistral. The legacy full-checkpoint
    path below is retained only for non-adapter directories.
    """
    import os as _os
    from src.prompts import MISTRAL_CHAT_TEMPLATE as _CHAT
    if _os.path.exists(_os.path.join(model_name, "adapter_config.json")):
        from src.model.loader import load_sara_for_eval
        model, tokenizer = load_sara_for_eval(
            model_name, device=device,
            dtype=torch.bfloat16, attn_implementation="sdpa", chat_template=_CHAT,
            hf_token=hf_token or _os.environ.get("HF_TOKEN"),
        )
        return model, tokenizer

    from transformers import BitsAndBytesConfig

    def update_vocab(tokenizer):
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [0, 1], \
                "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
            num_added_tokens += tokenizer.add_tokens([AddedToken(COMPRESS, lstrip=False, rstrip=False)])

        if 32001 in tokenizer.added_tokens_decoder:
            assert tokenizer.convert_ids_to_tokens([32001])[0] == COMPRESS

    num_added_tokens = 0
    if "checkpoint" in model_name:

        if os.path.basename(model_name).startswith("checkpoint-"):
            tokenizer_path = os.path.dirname(model_name)
        else:
            tokenizer_path = model_name

        MODEL_CLASS = AutoModelForCausalLM
        if "mistral" in model_name.lower():
            tokenizer_kwargs = {
                "padding_side": 'left',
                "add_eos_token": False,
                "use_fast": True,
                "local_files_only": True,
            }
        elif "llama" in model_name.lower():
            tokenizer_kwargs = {
                "padding_side": 'right',
                "local_files_only": True,
            }
        else:
            tokenizer_kwargs = {
                "token": hf_token,
                "local_files_only": True,
            }

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        update_vocab(tokenizer)
        
    elif "gemma" in model_name.lower():
        from transformers import Gemma3ForCausalLM, Gemma3Config
        # Original model
        MODEL_CLASS = Gemma3ForCausalLM
        tokenizer_kwargs = {
            "padding_side": 'right',
        }
        if "checkpoint" in model_name:
            tokenizer_kwargs["local_files_only"] = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        config_kwargs = {}
        if "checkpoint" in model_name:
            config_kwargs["local_files_only"] = True
        
        config = Gemma3Config.from_pretrained(model_name, **config_kwargs)

    else:
        MODEL_CLASS = AutoModelForCausalLM
        tokenizer_kwargs = {
            "padding_side": 'right',
        }
        if "checkpoint" in model_name:
            tokenizer_kwargs["local_files_only"] = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        

    if quantization is not None:

        if quantization in ["4bit", "8bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",  # Load model in 4-bit precision
                load_in_8bit=quantization == "8bit",
                bnb_4bit_quant_type="nf4",  # Normalize float 4 quantization
                bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
                bnb_4bit_use_double_quant=True  # Use nested quantization
            )

        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")

        # Add local_files_only=True only for checkpoint paths
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",  # Automatically choose best device
            "trust_remote_code": True,  # Required for some models
            "token": hf_token,
        }
        if "checkpoint" in model_name:
            model_kwargs["local_files_only"] = True

        model = MODEL_CLASS.from_pretrained(model_name, **model_kwargs)

    else:
        
        
        if "gemma" in model_name.lower():
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                # device_map="auto",
                "device_map": {"": 0} if device != "cpu" else "cpu",
                "token": hf_token,
            }
            if "checkpoint" in model_name:
                model_kwargs["local_files_only"] = True
            
            model = MODEL_CLASS.from_pretrained(model_name, **model_kwargs)
        
        else:
            # Load model with quantization
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                # device_map="auto",
                "device_map": {"": 0},
                "token": hf_token,
            }
            if "checkpoint" in model_name:
                model_kwargs["local_files_only"] = True
            
            model = MODEL_CLASS.from_pretrained(model_name, **model_kwargs)
        
        # Parameter: model.model.embed_tokens.weight

    if "checkpoint" in model_name:
        compress_token_id = tokenizer.convert_tokens_to_ids(COMPRESS)
        model.compress_token_id = compress_token_id

    return model, tokenizer


def _normalize(text):
    """
    Use Unicode normalization to decompose the characters in the input text string into their constituent components, specifically using the "Normalization Form D" (NFD) form.

    Example:
    - 'é' -> 'e' and '´'.
    - 'ñ' -> 'n' and '~'.

    """

    if isinstance(text, dict):
        assert 'answer' in text
        text = text['answer']
    else:
        assert isinstance(text, str)

    return unicodedata.normalize('NFD', text)


def get_substring_match_score(outputs, answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0], list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output, answer in zip(outputs, answers):
        if has_answer(answer, output):  # EM evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)

        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores) / len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return substring_match, substring_match_scores


def get_retrieval_embeds(model, input_ids, attention_mask=None, batch_size: int = 32):
    all_embeds = []

    if input_ids != []:
        with torch.no_grad():
            for index in range(0, input_ids.shape[0], batch_size):
                embeds = model.get_embedding(
                    input_ids=input_ids[index:(index + batch_size)].to(model.device),
                    attention_mask=attention_mask[index:(index + batch_size)].to(model.device), batch_size=batch_size
                )  # (B, D)
                all_embeds.append(embeds)
        embeds = torch.cat(all_embeds, dim=0)

    else:
        with torch.no_grad():
            embeds = torch.tensor([], dtype=torch.int64)
    # embeds = embeds.view(-1,embeds.shape[-1])
    return embeds

def get_retrieval_embeds_sentence_transformer(model, text, batch_size: int = 32):
    embeds = model.get_embedding(
        text, batch_size=batch_size
    )
    return embeds


@torch.no_grad()
def prepare_retrieval_embeds(backgrounds, retriever, tokenizer, batch_size=16):
    backgrounds = [backgrounds[idx:idx + batch_size] for idx in range(0, len(backgrounds), batch_size)]
    device = retriever.device
    ret = []
    for background in backgrounds:
        tokenized_retrieval_text = tokenizer(
            background,
            max_length=180,
            padding=True, truncation=True, return_tensors="pt")

        ## return a torch tensor of shape [batch_size,d_model]
        embeds = get_retrieval_embeds(
            model=retriever,
            input_ids=tokenized_retrieval_text['input_ids'].to(device),
            attention_mask=tokenized_retrieval_text['attention_mask'].to(device),
        ).cpu()

        embeds = [embeds[idx] for idx in range(embeds.shape[0])]
        ret.extend(embeds)
    return ret


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
            self,
            sequence: str,
            tokenizer: transformers.PreTrainedTokenizer,
            initial_decoder_input_length: int,
            batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length:]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len:]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


## copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/cb22e5028a6e40f409a539cbdd87194fd5e2570c/lm_eval/models/utils.py#L248

def stop_sequences_criteria(
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
        stop_sequences: List[str] = ['\n', '.'],
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def format_one_example(
        sample, include_answer, use_rag, retrieval_embed_length, task_type,
):
    question = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample['background']  ## a list
        background_prompts = ""

        for background in backgrounds:
            if retrieval_embed_length > 0:
                background_prompts += " ".join([COMPRESS] * retrieval_embed_length) + " "

            else:
                background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts)) + prompt

    return prompt, backgrounds


def get_n_shot_prompt(dev_data, n_shot, task_type, use_rag=False, retrieval_embed_length=0):
    assert n_shot >= 0, n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt, background = format_one_example(example, include_answer=True, use_rag=use_rag,
                                                    retrieval_embed_length=retrieval_embed_length, task_type=task_type)
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt, n_shot_background


def get_start_prompt(task_type, use_rag, sample=None):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False: "Answer the questions:"
        }[use_rag]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False: "Verify the following claims with \"True\" or \"False\":"
        }[use_rag]

    else:
        raise ValueError(f"Invalid task_type: {task_type}")


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def eval_truthfulqa(outputs, answers):
    f1_scores = []
    rl_scores = []
    for output, answer in zip(outputs, answers):
        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)

    return F1, RL, f1_scores, rl_scores


def eval_multiple_choice(generated_answers, answers):
    ret = []
    assert len(generated_answers) == len(answers)
    for g_answer, answer in zip(generated_answers, answers):
        ret.append(float(g_answer == answer))
    return round(sum(ret) / len(ret), 3), ret


def create_prompt_with_mistral_chat_format(messages, tokenizer, *args, **kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    formatted_text = ""
    for message in messages:
        if message['role'] == 'user':
            formatted_text += "[INST] " + message['content'] + " [/INST]"
        elif message['role'] == 'assistant':
            formatted_text += message['content'] + tokenizer.eos_token
        else:
            raise ValueError(
                "Mistral chat template only supports 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"])
            )
    # formatted_text += " The answer is:"
    return formatted_text


def load_compressor_and_tokenizers_eval(compressor_name_or_path: str, device: Union[torch.device, str]) -> tuple:
    from model import SFR
    if compressor_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
        compressor_kwargs = {
            "torch_dtype": torch.bfloat16,
        }
        if "checkpoint" in compressor_name_or_path:
            compressor_kwargs["local_files_only"] = True
        
        compressor = SFR.from_pretrained(compressor_name_or_path, **compressor_kwargs)
        
        tokenizer_kwargs = {}
        if "checkpoint" in compressor_name_or_path:
            tokenizer_kwargs["local_files_only"] = True
        
        compressor_tokenizer = AutoTokenizer.from_pretrained(compressor_name_or_path, **tokenizer_kwargs)
        compressor.eval()
        compressor = compressor.to(device)

        compressor_embed_length = compressor.get_embed_length()
        compressor_hidden_size = compressor.get_embed_dim()
        
    else:
        compressor = SentenceBERTEmbedding(
            args.compressor_name_or_path,
            torch_dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            # trust_remote_code=True,
            # token=os.environ['HF_TOKEN']
        )
        
        tokenizer_kwargs = {
            "token": os.environ['HF_TOKEN']
        }
        if "checkpoint" in args.compressor_name_or_path:
            tokenizer_kwargs["local_files_only"] = True
        
        compressor_tokenizer = AutoTokenizer.from_pretrained(args.compressor_name_or_path, **tokenizer_kwargs)

    return compressor, compressor_tokenizer, compressor_embed_length, compressor_hidden_size


def prepare_prompts(
        dev_data, test_data, task_type, tokenizer,
        n_shot=0, use_rag=False,
        retrieval_embed_length=0,
        chat_format=None,
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx, sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start = get_start_prompt(task_type, use_rag=use_rag, sample=sample)
            prompt_end, background = format_one_example(
                sample, include_answer=False, use_rag=use_rag, retrieval_embed_length=retrieval_embed_length,
                task_type=task_type)
            if 'subject' not in sample.keys():
                n_shot_prompt, n_shot_background = get_n_shot_prompt(dev_data, n_shot=n_shot, use_rag=use_rag,
                                                                     retrieval_embed_length=retrieval_embed_length,
                                                                     task_type=task_type)
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d['subject'] == sample['subject']:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects) == 5, sample['subject']
                n_shot_prompt, n_shot_background = get_n_shot_prompt(dev_data_with_same_subjects, n_shot=n_shot,
                                                                     use_rag=use_rag,
                                                                     retrieval_embed_length=retrieval_embed_length,
                                                                     task_type=task_type)

            if n_shot_prompt:
                prompt = prompt_start + splitter + splitter.join(n_shot_prompt) + splitter + prompt_end
            else:
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_format(messages, tokenizer) + " The answer is:"

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break

        prompts.append(prompt)
        backgrounds.append(background + n_shot_background)

    print("**" * 20, "show one example", "**" * 20)
    print(prompts[0])
    print("**" * 20, "show one example", "**" * 20)

    return prompts, backgrounds


@torch.no_grad()
def llm_for_open_generation(
        llm, llm_tokenizer,
        prompts,
        retrieval_embeds,
        batch_size=4,
        enable_progress_bar=True,
        task_type=None,

        max_new_tokens: int = 100
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx + batch_size] for idx in range(0, len(prompts), batch_size)]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx + batch_size] for idx in
                                    range(0, len(retrieval_embeds), batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)

    progress_bar = tqdm(range(total_test_number), ncols=60, disable=not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_prompt = llm_tokenizer(prompt, padding='longest', return_tensors='pt')
        input_ids = tokenized_prompt.input_ids.to(device)
        attention_mask = tokenized_prompt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs['retrieval_embeds'] = embeds
            stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])

        ## actual computation
        generated_output = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(generated_output[:, input_length:], skip_special_tokens=False)
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers


def prepare_retriever(args, documents, similarity_top_k=None):
    """Prepare retriever for document retrieval.

    Args:
        args: parsed args; reads args.chunk_size, args.retriever_name_or_path, args.n.
        documents: list of strings or pre-built llama_index Document objects.
        similarity_top_k: HF retriever top-k override. Defaults to args.n when None.
    """
    if documents and isinstance(documents[0], Document):
        doc_objects = documents
    else:
        text_splitter = SentenceSplitter(separator="\n", chunk_size=args.chunk_size, chunk_overlap=5)
        parsed_docs = []
        for doc in documents:
            parsed_docs.extend(text_splitter.split_text(doc))
        doc_objects = [Document(text=split, doc_id=f"{i}") for i, split in enumerate(parsed_docs)]

    retriever_path = args.retriever_name_or_path if args.retriever_name_or_path else "bm25"

    if retriever_path == "bm25":
        retriever, prepare_time = getBM25Retriever(
            doc_objects, similarity_top_k=len(doc_objects), chunk_size=args.chunk_size
        )
    elif retriever_path in ["Salesforce/SFR-Embedding-Mistral", "BAAI/bge-reranker-v2-m3"]:
        if not isinstance(Settings.embed_model, HuggingFaceEmbedding) or Settings.embed_model is None:
            Settings.embed_model = HuggingFaceEmbedding(model_name=retriever_path)
        top_k = similarity_top_k if similarity_top_k is not None else args.n
        retriever, prepare_time = getHuggingFaceRetriever(doc_objects, similarity_top_k=top_k)
    else:
        raise ValueError(f"Unsupported retriever type: {retriever_path}")

    print_colored(f"Retriever preparation time: {prepare_time:.2f}s", "cyan")
    return retriever

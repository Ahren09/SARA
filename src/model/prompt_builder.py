# prompt_builder.py
from src.prompts import (
    PROMPT_HF_MODELS,
    PROMPT_HF_MODELS_QUESTION_ONLY,
    ANSWER_INSTRUCTION,
    ANSWER_INSTRUCTION_MULTI_CHOICE,
    PROMPT_MISTRAL
)
from nltk.tokenize import sent_tokenize

from src.const import COMPRESS


def build_prompt(question, context, additional_context, args, choices=None):
    context_str, additional_prompt = "", ""
    context_list = []
    additional_context_new = []

    if isinstance(context, list):
        context_list = context
        context_str = "\n\n".join(context)
    elif isinstance(context, str):
        context_list = [context]
        context_str = context

    if additional_context:
        for i, ad in enumerate(additional_context):
            if args.compressor_encode_method == "sentence":
                sents = sent_tokenize(ad)
                additional_prompt += f"Document {i+1}. {COMPRESS * len(sents)}\n"
                additional_context_new.extend(sents)
            elif args.compressor_encode_method == "paragraph":
                additional_prompt += f"Document {i+1}. {COMPRESS}\n"
                additional_context_new.append(ad)
            else:
                raise ValueError(f"Unsupported compressor encode method: {args.compressor_encode_method}")

    if not context_str:
        return PROMPT_HF_MODELS_QUESTION_ONLY.format(
            question=question,
            answer_instruction=ANSWER_INSTRUCTION
        ), context_list, additional_context_new

    if args.answer_format == "multiple_choice" and choices:
        question_fmt = f"{question}\n" + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return PROMPT_HF_MODELS.format(
            question=question_fmt,
            context=context_str,
            answer_instruction=ANSWER_INSTRUCTION_MULTI_CHOICE
        ), context_list, additional_context_new

    if "mistral" in args.model_name_or_path:
        return PROMPT_MISTRAL.format(
            question=question,
            context=context_str,
            additional_context=additional_prompt,
            answer_instruction=ANSWER_INSTRUCTION
        ), context_list, additional_context_new

    return PROMPT_HF_MODELS.format(
        question=question,
        context=context_str,
        answer_instruction=ANSWER_INSTRUCTION
    ), context_list, additional_context_new

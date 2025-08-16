import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import re

import torch
from transformers import PreTrainedTokenizerFast


def number_parser(text):
    """
    use re to check for digits
    use w2n to convert to numbers
    return the number
    Parameters
    ----------
    text: str
        the text to convert to a number
    Returns
    -------
    float
        the number converted from the text
    """

    from word2number import w2n

    _re = re.findall(r"[+-]?(\d*\.\d+|\d+)", text)
    if len(_re) != 0:
        if len(_re) > 1:
            logging.info(
                f"More than one number found in {text}. Overriding error and selecting the first."
            )
        return float(_re[0])

    try:
        return w2n.word_to_num(text)
    except ValueError:
        logging.info("ValueError")
        logging.info(
            f"Error override. Could not convert to number, "
            f"setting to -1. text: {text}"
        )
    return -1


def confidence_in_words(
    o, tokenizer: PreTrainedTokenizerFast = None, device: str = "cpu"
):
    if tokenizer is not None:
        out_text = tokenizer.batch_decode(o.sequences)
    else:
        out_text = o
    return torch.tensor([number_parser(t.split("A:")[-1]) for t in out_text]).to(device)


def sequence_log_score(logits, input_toks, pad_token_id=0):
    # get the logit score for the whole statement
    logits = logits[:, :-1, :]  # (batch, seq_len, vocab_size)
    # would this also work without logsoftmax?
    log_softs = logits.log_softmax(-1)
    # select the logit score corresponding to the input sentence
    input_log_softs = torch.gather(
        log_softs, 2, input_toks[:, 1:, None]).squeeze(-1)
    # mask to replace padding tokens by identity multiplications
    mask = input_toks[:, 1:] == pad_token_id
    masked_log_softs = input_log_softs.masked_fill(mask, 1.0)
    # compute the probability of the whole statement
    sentence_log_softs = masked_log_softs.mean(-1)
    return sentence_log_softs.exp()


def surrogate_logit_score(o, targets, pad_token_id=0):
    limit = min(targets.shape[1], len(o.scores))
    logits = torch.stack(o.scores[-limit:])
    probs = logits.log_softmax(-1)
    tokens_probs = probs.swapaxes(0, 1).gather(
        2, targets[:, -limit:].swapaxes(0, 1).repeat(probs.shape[1], 1, 1)
    )

    return tokens_probs


# Test each scoring method


def test_sequence_log_score():
    """
    Test sequence log score for GPT-2 outputs.
    """
    tokenized_input = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_output = model(**tokenized_input)
    scores = sequence_log_score(
        model_output,
        input_toks=tokenized_input["input_ids"],
        pad_token_id=tokenizer.pad_token_id,
    )
    print("Sequence Log Scores:", scores)


def test_surrogate_logit_score():
    """
    Test surrogate logit score for GPT-2 outputs.
    """
    surrogate_targets = tokenizer(
        ["Yes", "No", "Maybe"], padding=True, truncation=True, return_tensors="pt"
    )["input_ids"].to(device)
    tokenized_input = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_output = model.generate(
            **tokenized_input,
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    scores = surrogate_logit_score(model_output, targets=surrogate_targets)
    print("Surrogate Logit Scores:", scores)


def test_confidence_in_words():
    """
    Test confidence in verbalized words for GPT-2 outputs.
    """
    tokenized_input = tokenizer(
        batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_output = model.generate(
            **tokenized_input,
            max_new_tokens=20,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    scores = confidence_in_words(
        model_output, tokenizer=tokenizer, device=device)
    print("Confidence in Words:", scores)


if __name__ == "__main__":
    # Load GPT-2 model and tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "gpt2"

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a new padding token
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    # Resize model embeddings to account for the new token
    model.resize_token_embeddings(len(tokenizer))


    # Define input prompts
    batch = ["The sky is blue.", "Cats are better than dogs."]  # Example prompts

    print("Testing sequence_log_score:")
    test_sequence_log_score()
    print("\nTesting surrogate_logit_score:")
    test_surrogate_logit_score()
    print("\nTesting confidence_in_words:")
    test_confidence_in_words()

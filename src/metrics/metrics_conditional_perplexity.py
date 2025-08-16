
import evaluate
import torch
from transformers import AutoTokenizer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ConditionalSelfInformation:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def compute_log_probability(self, texts):
        if isinstance(texts, str):  # Ensure input is always a list
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        seq_lens = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
        batch_log_probs = []

        for i in range(len(texts)):
            ids = input_ids[i, 1:seq_lens[i]]
            token_log_probs = log_probs[i, :seq_lens[i] - 1].gather(1, ids.unsqueeze(-1)).squeeze(-1)
            batch_log_probs.append(token_log_probs.sum().item())

        return batch_log_probs

    def compute_conditional_self_information(self, docs_A, doc_B):
        if isinstance(doc_B, str):
            # Single `doc_B`, concatenate with all `docs_A`
            docs_A_given_B = [doc_B + " " + doc_A for doc_A in docs_A]
        elif isinstance(doc_B, list):
            if len(doc_B) != len(docs_A):
                raise ValueError("Length of doc_B list must match length of docs_A list.")
            # Pairwise concatenation
            docs_A_given_B = [b + " " + a for a, b in zip(docs_A, doc_B)]
        else:
            raise TypeError("doc_B must be either a string or a list of strings.")

        # Compute log probabilities
        log_prob_A = self.compute_log_probability(docs_A)
        log_prob_A_given_B = self.compute_log_probability(docs_A_given_B)

        # Compute conditional self-information
        conditional_self_info = [-(p_given_B - p_A) for p_given_B, p_A in zip(log_prob_A_given_B, log_prob_A)]
        return conditional_self_info


# Archived code
class ConditionalPerplexityEvaluator:
    def __init__(self, model_id="gpt2", device="cpu"):
        self.perplexity = evaluate.load("perplexity", module_type="metric")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device

    def get_token_length(self, text):
        return len(self.tokenizer.encode(text))

    def get_ppl(self, text):
        results = self.perplexity.compute(model_id="gpt2", add_start_token=False, predictions=[text])
        return results["perplexities"][0]

    def get_conditional_ppl(self, text, question, condition_in_question="none"):
        if condition_in_question == "none":
            return self.get_ppl(text)
        elif condition_in_question == "before":
            conditioned_text = question + " " + text
            return self.get_ppl(conditioned_text)
        elif condition_in_question == "after":
            conditioned_text = text + " " + question
            return self.get_ppl(conditioned_text)
        else:
            raise ValueError("Invalid condition_in_question value. Use 'none', 'before', or 'after'.")

if __name__ == "__main__":
    # Example usage
    model = ConditionalSelfInformation(model_name="gpt2", device=args.device)

    docs_A = [
        "This is a new discovery.",
        "The experiment yielded unexpected results.",
        "Scientists found a new way to harness solar energy."
    ]
    doc_B = "Scientists have made an important finding."

    conditional_info = model.compute_conditional_self_information(docs_A, doc_B)

    for i, info in enumerate(conditional_info):
        print(f"Conditional Self-Information of A[{i}] given B: {info}")


    
    # Archived: Example usage for ConditionalPerplexityEvaluator
    evaluator = ConditionalPerplexityEvaluator()
    text = "The weather is nice today."
    question = "What is the weather like?"

    print("Unconditional PPL:", evaluator.get_conditional_ppl(text, question, "none"))
    print("Conditioned on question before:", evaluator.get_conditional_ppl(text, question, "before"))
    print("Conditioned on question after:", evaluator.get_conditional_ppl(text, question, "after"))




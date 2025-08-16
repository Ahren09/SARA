import torch
from llama_index.core import Document
from tqdm import tqdm
from transformers import AutoTokenizer

from src.arguments import parse_args
from src.model import SFR
from src.model.retriever import getBM25Retriever, getGeminiRetriever, getOpenAIRetriever, getHuggingFaceRetriever
from src.utils.data_utils import reformat_dataset_multi_round as reformat_dataset, load_dataset_for_eval
from src.utils.eval_utils import load_eval_model, format_prompt, get_retrieval_embeds
from src.utils.utility import print_colored


def main():
    dataset = load_dataset_for_eval(args)
    reformatted_dataset = reformat_dataset(dataset, args)

    model, tokenizer = load_eval_model(args.model_name_or_path, quantization=args.quantization)

    if "checkpoint" in args.model_name_or_path.lower():
        compressor = SFR.from_pretrained(args.embedding_model, torch_dtype=torch.bfloat16)
        compressor.to(args.device)
        compressor.eval()
        compressor_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    else:
        compressor = compressor_tokenizer = None

    for example_id, example in enumerate(tqdm(reformatted_dataset, desc=args.dataset_name)):
        contexts = [paragraph['context'] for paragraph in example['paragraphs']]
        documents = [Document(doc_id=str(doc_id), text=t) for (doc_id, t) in enumerate(contexts)]

        if compressor is not None:
            tokenized_retrieval_text = compressor_tokenizer(
                contexts,
                max_length=args.max_seq_length,
                padding=True, truncation=True, return_tensors="pt"
            )

            # TODO: This retrieval_embeds can be prepared before the loop
            retrieval_embeds = get_retrieval_embeds(
                model=compressor,
                input_ids=tokenized_retrieval_text['input_ids'],
                attention_mask=tokenized_retrieval_text['attention_mask'],
            )

        # Initialize sparse retriever
        if args.retriever_name_or_path == "bm25":
            sparse_retriever, prepare_time = getBM25Retriever(documents, similarity_top_k=args.k)

        # Initialize dense retriever
        if args.retriever_name_or_path == "gemini":
            dense_retriever, prepare_time = getGeminiRetriever(documents, similarity_top_k=args.k)
        elif args.retriever_name_or_path == "openai":
            dense_retriever, prepare_time = getOpenAIRetriever(documents, similarity_top_k=args.k)


        else:

            # Use a HuggingFace model as the retriever
            # BAAI models are generally good for this
            dense_retriever, prepare_time = getHuggingFaceRetriever(documents,
                                                                    model_name_or_path=args.retriever_name_or_path,
                                                                    similarity_top_k=args.k)

        if len(example['paragraphs']) == 0:
            continue

        # Iterate through each paragraph
        for paragraph_id, paragraph_dict in enumerate(example['paragraphs']):

            # Iterate through each question

            for question_id, question_dict in enumerate(paragraph_dict['qas']):
                question = question_dict['question']

                retrieval_kwargs = {}

                # TODO: Add dense retriever
                nodes = sparse_retriever.retrieve(question)

                selected_doc_ids = [node.node.ref_doc_id for node in nodes]
                selected_doc_ids = [int(doc_id) for doc_id in selected_doc_ids][1:]

                if compressor is not None:
                    retrieval_kwargs['retrieval_embeds'] = retrieval_embeds[selected_doc_ids]

                knowledge = "\n---\n".join([node.text for node in nodes[:1]])
                additional_context = [node.text for node in nodes[1:]]

                prompt = format_prompt(args.model_name_or_path, question, knowledge, additional_context)

                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                output = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,  # Set the maximum length of the generated text
                    do_sample=False,  # Ensures greedy decoding,
                    temperature=min(args.temperature, 0.001),
                    use_cache=True,
                    **retrieval_kwargs
                )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

                # generated_text = generated_text[generated_text.find(question) + len(question):]

                # assert "Question:" in generated_text, f"Generated text does not contain 'Question:'"
                # assert "Assistant:" in generated_text, f"Generated text does not contain 'Answer:'"

                generated_text = generated_text.split("Question:")[-1].split("Assistant:")[-1].split("Answer")[
                    -1].strip()

                # print("R: ", knowledge)
                print_colored(f"Q:\t{question}", "red")
                print_colored(f"A:\t{generated_text}", "yellow")
                print_colored(f"GT:\t{question_dict['answers'][0]['text']}", "blue")


if __name__ == "__main__":
    args = parse_args('test')
    main()

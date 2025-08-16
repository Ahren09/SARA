import torch

def collator(        
        samples,
        llm_tokenizer,
        retriever_tokenizer = None,
        retrieval_context_length = 180,
        task="pretrain",
    ):
    """
    collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict): a dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer: tokenizer for llm
        retriever_tokenizer: tokenizer for retriever
        retrieval_context_length: max length for the retrieved passages
    
    Returns:
        input_ids: input_ids with compress_token_id (labels,attention_mask)
        input_ids: input_ids for llm without compress_token_id (labels,attention_mask)
        compressor_input_ids: input_ids for retriever (compressor_attention_mask)

    """
    def padding(input_ids,labels=None,padding_side='right'):
        """
        batch padding
        """

        def _padding(ids,padding_value,padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  

                # Ahren: added the `padding_value` argument
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids,batch_first=True, padding_value=padding_value),
                    dims=[1],
                )

                # return torch.flip(
                #     torch.nn.utils.rnn.pad_sequence(flipped_ids,batch_first=True,padding_value=padding_value),
                #     dims=[1],
                # )
                
        pad_token_id = llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
        input_ids = _padding(input_ids,padding_value=pad_token_id,padding_side=padding_side)
        attention_mask = (input_ids != pad_token_id).long()
        if labels is not None:
            labels = _padding(labels,padding_value=-100,padding_side=padding_side)
            
            if "llama" in llm_tokenizer.name_or_path.lower():
                # TODO: check if this setting (for llama only) is correct
                labels[input_ids == pad_token_id] = -100
        return input_ids,attention_mask,labels

    if isinstance(samples,dict):
        samples = [samples]

    input_ids,attention_mask,labels = padding(
        input_ids=[x['input_ids'] for x in samples],
        labels=[x['labels'] for x in samples] if 'labels' in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )
    
    
    
    assert task in ['pretrain', 'paragraph', "finetune", "finetune_stage1"]

    ret = {
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "labels":labels,
        
    }
    
    if 'input_ids_nl' in samples[0]:
        input_ids_nl, attention_mask_nl, labels_nl = padding(
            input_ids=[x['input_ids_nl'] for x in samples],
            labels=[x['labels_nl'] for x in samples] if 'labels' in samples[0].keys() else None,
            padding_side=llm_tokenizer.padding_side
        )
        
        ret['input_ids_nl'] = input_ids_nl
        ret['attention_mask_nl'] = attention_mask_nl
        ret['labels_nl'] = labels_nl
        

    if 'retriever_input_text' in samples[0].keys():

        # Records which compression tokens belong to which example
        retriever_example_ids = []
        for i, x in enumerate(samples):
            retriever_example_ids.extend([i]*len(x['retriever_input_text']))

        retriever_input_text = [x['retriever_input_text'] for x in samples]
        if task in ['pretrain', 'paragraph', "finetune"]:
            assert isinstance(retriever_input_text[0],list)
            retriever_input_text = [x for y in retriever_input_text for x in y]
            
        elif task == 'finetune_stage2':
            assert isinstance(retriever_input_text[0],str)
            
            
        
        ## handling different retriever tokenization problem
        if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
            retriever_input_text = ["passage: "+x for x in retriever_input_text]
        elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
            retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]

        # Remove placeholder retrieval text
        retriever_input_text = [text for text in retriever_input_text if text]

        if len(retriever_input_text) > 0:
            tokenized_retrieval_text = retriever_tokenizer(
                retriever_input_text, 
                max_length=retrieval_context_length,
                padding=True, truncation=True, return_tensors="pt"
            )
            
            ret['compressor_input_ids']      = tokenized_retrieval_text['input_ids']
            ret['compressor_attention_mask'] = tokenized_retrieval_text['attention_mask']
            
        else:
            ret['compressor_input_ids'] = []
            ret['compressor_attention_mask'] = []
            
            
        ret['retriever_example_ids'] = retriever_example_ids

    """
    if 'input_ids' in samples[0].keys():
        input_ids = [x['input_ids'] for x in samples]
        labels =    [x['labels'] for x in samples]
     
        input_ids,attention_mask,labels = padding(input_ids,labels,padding_side=llm_tokenizer.padding_side)
        
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask
        ret['labels'] = labels

    """
    return ret


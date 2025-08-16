import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, MistralModel
from sentence_transformers import SentenceTransformer


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]) # Only if all sequences are of the same length
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class MLPClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim2=256):
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim2),     
            nn.GELU(),
            nn.Linear(hidden_dim2, 2)         
        )

    def forward(self, query_embed, doc_embed):
        query_embed = query_embed.unsqueeze(1).expand(-1, doc_embed.size(1), -1)
        assert query_embed.size() == doc_embed.size()
        inputs = torch.cat([query_embed, doc_embed], dim=-1)
        logits = self.mlp(inputs)
        return logits

class SFR(MistralModel):

    def get_embed_dim(self):
        return self.config.hidden_size
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self,input_ids,attention_mask, batch_size:int=64):
        outputs = self.forward(input_ids=input_ids,attention_mask=attention_mask) # (B, L, 4096)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings

class SentenceBERTEmbedding(nn.Module):
    def __init__(self, model_name_or_path, torch_dtype, device):
        super(SentenceBERTEmbedding, self).__init__()
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, 
                                         device=device, 
                                         trust_remote_code=True,
                                         model_kwargs={"torch_dtype":torch_dtype})
        
        self.model.to(self.device)
        
    def get_embed_dim(self):
        return self.model.get_sentence_embedding_dimension()
    
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self, text, batch_size=64):
        outputs = self.model.encode(text, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        
        return outputs

class SFRReranker(nn.Module):
    def __init__(self, model, tokenizer, **kwargs):
        super(SFRReranker, self).__init__()
        self.batch_size = kwargs.get("batch_size", 32)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.embed_model = model
        self.tokenizer = tokenizer
        self.mlp = MLPClassifier(self.embed_model.get_embed_dim() * 2, 128)
        self.embed_model.to(self.device)
        self.mlp.to(self.device)
        
        self.loss_fct = nn.CrossEntropyLoss()
        
    
    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask, labels, **kwargs):
        
        # (B, D)
        query_embeds = self.embed_model.get_embedding(input_ids=query_input_ids, attention_mask=query_attention_mask)
        B, N, L = doc_input_ids.shape
        doc_embeds = self.embed_model.get_embedding(input_ids=doc_input_ids.view(-1, L), attention_mask=doc_attention_mask.view(-1, L))
        doc_embeds = doc_embeds.view(B, N, -1)

        """
        
        doc_input_ids = doc_input_ids.view(-1, D)
        doc_attention_mask = doc_attention_mask.view(-1, D)
        
        doc_embeds = []
        for i in range(0, doc_input_ids.shape[0], self.batch_size):
            batch_embed = self.embed_model.get_embedding(
                input_ids=doc_input_ids[i : (i + self.batch_size)],
                attention_mask=doc_attention_mask[i : (i + self.batch_size)]
            )
            doc_embeds.append(batch_embed)
        doc_embeds = torch.cat(doc_embeds, dim=0)  # Concatenate instead of stacking
        """
        class_counts = torch.bincount(labels.view(-1))
        total_samples = labels.numel()
        class_weights = total_samples / (2.0 * class_counts)  # Balance formula

        # Normalize the weights to avoid scaling issues
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(self.device)

        # Define the CrossEntropyLoss with dynamic weights
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        
        
        logits = self.mlp(query_embeds, doc_embeds)  # Shape: (batch_size, num_docs, 2)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))  # Flatten for loss computation
        pred = logits.argmax(dim=-1)
        return {"loss": loss, "pred": pred}
        
        

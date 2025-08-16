import torch

def cosine_distance_torch(query, embedding, return_tensor=False):
    """
    Case 1:
    - query: (1, D)
    - embedding: (D)

    Case 2:
    - query: (1, D)
    - embedding: (N, D)
    """

    if embedding.dim() == 1:
        if query.dim() == 2: 
            assert query.shape[0] == 1
            query = query.squeeze(0)
        query_norm = torch.nn.functional.normalize(query, dim=0)
        embedding_norm = torch.nn.functional.normalize(embedding, dim=0)
        sim = 1 - torch.dot(query_norm, embedding_norm)
        if return_tensor:
            return sim
        return sim.item()

    elif embedding.dim() == 2:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        assert query.shape[0] == 1
        query_norm = torch.nn.functional.normalize(query, dim=1)
        embedding_norm = torch.nn.functional.normalize(embedding, dim=1)
        sim = 1 - torch.matmul(query_norm, embedding_norm.T)
        if return_tensor:
            return sim.squeeze()
        return sim.squeeze().tolist()

def l1_distance_torch(query, embedding):
    return torch.sum(torch.abs(query - embedding))

def l2_distance_torch(query, embedding):
    return torch.sqrt(torch.sum((query - embedding) ** 2))

def linf_distance_torch(query, embedding):
    return torch.max(torch.abs(query - embedding))

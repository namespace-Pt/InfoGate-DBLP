import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel
from transformers import AutoModel



class GraphSage(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)
        self.plm_dim = manager.plm_dim

        self.plm = AutoModel.from_pretrained(manager.plm_dir)
        self.plm.pooler = None
        self.graph_transform = nn.Linear(manager.plm_dim * 2, manager.plm_dim, bias=False)
        self.pooling_transform = nn.Linear(manager.plm_dim, manager.plm_dim)


    def aggregation(self, neighbor_embed, neighbor_mask):
        neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
        neighbor_embed = neighbor_embed * neighbor_mask.unsqueeze(-1)
        return torch.max(neighbor_embed, dim=-2)[0]


    def graphsage(self, node_embed, node_mask):
        neighbor_embed = node_embed[:, 1:]  # B N D
        neighbor_mask = node_mask[:, 1:]  # B N
        center_embed = node_embed[:, 0]  # B D
        neighbor_embed = self.aggregation(neighbor_embed, neighbor_mask)  # B D
        main_embed = torch.cat([center_embed, neighbor_embed], dim=-1)  # B 2D
        main_embed = self.graph_transform(main_embed)
        main_embed = F.relu(main_embed)
        return main_embed


    def _encode_node(self, token_id, attn_mask, neighbor_mask):
        B, N, L = token_id.shape
        # D = self.config.hidden_size
        token_id = token_id.view(-1, L)
        attn_mask = attn_mask.view(B * N, L)
        embeddings = self.plm(token_id, attention_mask=attn_mask).last_hidden_state[:, 0].view(B, N, self.plm_dim)
        node_embeddings = self.graphsage(embeddings, neighbor_mask)
        return node_embeddings


    def infer(self, x):
        query_token_id = x["query_token_id"].to(self.device, non_blocking=True)
        key_token_id = x["key_token_id"].to(self.device, non_blocking=True)
        query_attn_mask = x['query_attn_mask'].to(self.device, non_blocking=True)
        key_attn_mask = x['key_attn_mask'].to(self.device, non_blocking=True)
        query_neighbor_mask = x["query_neighbor_mask"].to(self.device, non_blocking=True)
        key_neighbor_mask = x["key_neighbor_mask"].to(self.device, non_blocking=True)

        query_embedding = self._encode_node(query_token_id, query_attn_mask, query_neighbor_mask)
        key_embedding = self._encode_node(key_token_id, key_attn_mask, key_neighbor_mask)

        score = query_embedding.matmul(key_embedding.transpose(-2,-1))
        return score


    def forward(self, x):
        score = self.infer(x)  # B, B
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(score, labels)
        return loss

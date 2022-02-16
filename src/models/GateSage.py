import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel
from .modules.encoder import GatedBertEncoder
from .modules.weighter import *



class GateSage(BaseModel):
    def __init__(self, manager, weighter):
        super().__init__(manager)
        self.plm_dim = manager.plm_dim

        self.encoder = GatedBertEncoder(manager)
        self.weighter = weighter
        self.graph_transform = nn.Linear(manager.plm_dim * 2, manager.plm_dim, bias=False)
        self.pooling_transform = nn.Linear(manager.plm_dim, manager.plm_dim)

        self.k = manager.k
        keep_k_modifier = torch.zeros(manager.sequence_length)
        keep_k_modifier[1:self.k + 1] = 1
        self.register_buffer('keep_k_modifier', keep_k_modifier, persistent=False)


    def _compute_gate(self, token_id, attn_mask, gate_mask, token_weight):
        """ gating by the weight of each token

        Returns:
            gated_token_ids: [B, K]
            gated_attn_masks: [B, K]
            gated_token_weight: [B, K]
        """
        if gate_mask is not None:
            keep_k_modifier = self.keep_k_modifier * (gate_mask.sum(dim=-1, keepdim=True) < self.k)
            pad_pos = ~((gate_mask + keep_k_modifier).bool())   # B, L
            token_weight = token_weight.masked_fill(pad_pos, -float('inf'))

            gated_token_weight, gated_token_idx = token_weight.topk(self.k)
            gated_token_weight = torch.softmax(gated_token_weight, dim=-1)
            gated_token_id = token_id.gather(dim=-1, index=gated_token_idx)
            gated_attn_mask = attn_mask.gather(dim=-1, index=gated_token_idx)
            # gated_gate_mask = gate_mask.gather(dim=-1, index=gated_token_idx)

        # heuristic gate
        else:
            if token_id.dim() == 2:
                gated_token_id = token_id[:, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, 1: self.k + 1]
            else:
                gated_token_id = token_id[:, :, 1: self.k + 1]
                gated_attn_mask = attn_mask[:, :, 1: self.k + 1]
            gated_token_weight = None
        return gated_token_id, gated_attn_mask, gated_token_weight


    def _encode_node(self, token_id, attn_mask, gate_mask, neighbor_mask):
        token_weight = self.weighter(token_id, attn_mask)
        gated_token_id, gated_attn_mask, gated_token_weight = self._compute_gate(token_id, attn_mask, gate_mask, token_weight)
        embedding = self.encoder(gated_token_id, gated_attn_mask, gated_token_weight)
        node_embedding = self.graphsage(embedding, neighbor_mask)
        return node_embedding


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


    def infer(self, x):
        query_token_id = x["query_token_id"].to(self.device, non_blocking=True)
        key_token_id = x["key_token_id"].to(self.device, non_blocking=True)
        query_attn_mask = x['query_attn_mask'].to(self.device, non_blocking=True)
        key_attn_mask = x['key_attn_mask'].to(self.device, non_blocking=True)
        query_neighbor_mask = x["query_neighbor_mask"].to(self.device, non_blocking=True)
        key_neighbor_mask = x["key_neighbor_mask"].to(self.device, non_blocking=True)
        try:
            query_gate_mask = x["query_gate_mask"].to(self.device, non_blocking=True)
            key_gate_mask = x["key_gate_mask"].to(self.device, non_blocking=True)
        except:
            query_gate_mask = None
            key_gate_mask = None

        query_embedding = self._encode_node(query_token_id, query_attn_mask, query_gate_mask, query_neighbor_mask)
        key_embedding = self._encode_node(key_token_id, key_attn_mask, key_gate_mask, key_neighbor_mask)

        score = query_embedding.matmul(key_embedding.transpose(-2,-1))
        return score


    def forward(self, x):
        score = self.infer(x)  # B, B
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(score, labels)
        return loss

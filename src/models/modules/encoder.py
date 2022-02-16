import torch
import torch.nn as nn
from transformers import AutoModel
from .attention import scaled_dp_attention, extend_attention_mask, TFMLayer



class GatedBertEncoder(nn.Module):
    def __init__(self, manager):
        super().__init__(manager)
        self.name = type(self).__name__[:-11]
        plm = AutoModel.from_pretrained(manager.plm_dir)
        self.embeddings = plm.embeddings
        self.plm = plm.encoder

        self.news_query = nn.Parameter(torch.randn((1, manager.hidden_dim), requires_grad=True))
        nn.init.xavier_normal_(self.news_query)
        # self.newsProject = nn.Linear(manager.hidden_dim, manager.hidden_dim)
        # nn.init.xavier_normal_(self.newsProject.weight)
        # self.Tanh = nn.Tanh()


    def forward(self, token_id, attn_mask, token_weight=None):
        original_shape = token_id.shape
        token_id = token_id.view(-1, original_shape[-1])
        attn_mask = attn_mask.view(-1, original_shape[-1])

        token_embedding = self.embeddings(token_id)

        if token_weight is not None:
            token_weight = token_weight.view(-1, original_shape[-1])
            token_embedding = token_embedding * token_weight.unsqueeze(-1)

        extended_attn_mask = extend_attention_mask(attn_mask)
        token_embedding = self.plm(token_embedding, attention_mask=extended_attn_mask).last_hidden_state
        # we do not keep [CLS] and [SEP] after gating, so it's better to use attention pooling
        embedding = scaled_dp_attention(self.news_query, token_embedding, token_embedding, attn_mask=attn_mask.unsqueeze(-2)).squeeze(dim=-2).view(*original_shape[:-1], -1)
        return embedding


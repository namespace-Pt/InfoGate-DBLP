import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from utils.util import roc_auc_score, mrr_score, ndcg_score



class BaseModel(nn.Module):
    def __init__(self, manager, name=None):
        super().__init__()

        self.sequence_length = manager.sequence_length
        self.device = manager.device
        self.rank = manager.rank
        self.world_size = manager.world_size

        self.enable_gate = manager.enable_gate
        self.k = manager.k

        self.neighbor_num = manager.neighbor_num

        if name is None:
            name = type(self).__name__
        if manager.verbose is not None:
            self.name = "-".join([name, manager.verbose])
        else:
            self.name = name

        self.crossEntropy = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(self.name)


    def _gather_tensors(self, local_tensor):
        """
        gather tensors from all gpus

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            all_tensors: concatenation of local_tensor in each process
        """
        all_tensors = [torch.empty_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, local_tensor)
        all_tensors[self.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)


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


    def get_optimizer(self, manager, dataloader_length):
        optimizer = optim.Adam(self.parameters(), lr=manager.learning_rate)

        scheduler = None
        if manager.scheduler == "linear":
            total_steps = dataloader_length * manager.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = round(manager.warmup * total_steps),
                                            num_training_steps = total_steps)

        return optimizer, scheduler


    def _eval(self, manager, loader):
        metrics = defaultdict(list)
        for i, x in enumerate(tqdm(loader, desc="Evaluating", ncols=80)):
            score = self.infer(x)
            label = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=self.device)

            predictions = torch.argmax(score, dim=-1)
            acc = (torch.sum((predictions == label)) / label.shape[0]).item()

            score = score.cpu().numpy()
            label = torch.eye(label.shape[0]).numpy()
            auc_all = [roc_auc_score(label[i], score[i]) for i in range(label.shape[0])]
            mrr_all = [mrr_score(label[i], score[i]) for i in range(label.shape[0])]
            ndcg_all = [ndcg_score(label[i], score[i], label.shape[1]) for i in range(label.shape[0])]

            metrics["acc"].append(acc)
            metrics["auc"].append(np.mean(auc_all))
            metrics["mrr"].append(np.mean(mrr_all))
            metrics["ndcg"].append(np.mean(ndcg_all))

        for k,v in metrics.items():
            metrics[k] = np.asarray(v).mean()
        return dict(metrics)


    @torch.no_grad()
    def dev(self, manager, loaders, load=True, log=False):
        self.eval()
        if load:
            manager.load(self)

        if self.rank in [0, -1]:
            metrics = self._eval(manager, loaders["dev"])
            metrics["main"] = metrics["acc"]
            self.logger.info(metrics)
            if log:
                manager._log(self.name, metrics)
        else:
            metrics = None

        if manager.distributed:
            dist.barrier(device_ids=[self.device])

        return metrics


    @torch.no_grad()
    def test(self, manager, loaders, load=True, log=False):
        self.eval()
        if load:
            manager.load(self)

        if self.rank in [0, -1]:
            metrics = self._eval(manager, loaders["test"])
            metrics["main"] = metrics["acc"]
            self.logger.info(metrics)
            if log:
                manager._log(self.name, metrics)
        else:
            metrics = None

        if manager.distributed:
            dist.barrier(device_ids=[self.device])

        return metrics


    @torch.no_grad()
    def inspect(self, manager, loaders):
        assert manager.enable_gate != "none"

        tokenizer = AutoTokenizer.from_pretrained(manager.plm_dir)
        loader_dev = loaders["dev"]
        for i, x in enumerate(loader_dev):
            token_ids = x["query_token_id"].to(self.device)
            attn_masks = x['query_attn_mask'].to(self.device)
            gate_masks = x['query_gate_mask'].to(self.device)
            for token_id, attn_mask, gate_mask in zip(token_ids, attn_masks, gate_masks):
                token_weight = self.weighter(token_id, attn_mask)
                gated_token_ids, gated_attn_masks, gated_token_weights = self._compute_gate(token_id, attn_mask, gate_mask, token_weight)
                for tid, gated_token_id, gated_token_weight in zip(token_id.tolist(), gated_token_ids.tolist(), gated_token_weights.tolist()):
                    gated_token = tokenizer.convert_ids_to_tokens(gated_token_id)
                    print("-"*10 + "original text" + "-"*10)
                    print(tokenizer.decode(tid))
                    print("-"*10 + "gated tokens" + "-"*10)
                    line = "; ".join([f"{i} ({round(p, 3)})" for i, p in zip(gated_token, gated_token_weight)])
                    print(line)
                    input()


    @torch.no_grad()
    def encode(self ,manager, loaders):
        """used for testing speed"""
        for i, x in enumerate(tqdm(loaders["dev"], desc="Evaluating", ncols=80)):
            self.infer(x)

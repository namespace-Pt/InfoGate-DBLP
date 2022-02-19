import os
import logging
import linecache
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from itertools import cycle
from multiprocessing import Pool
from transformers import AutoTokenizer
from torch.utils.data import Dataset

logger = logging.getLogger("Dataset")


class DBLP(Dataset):
    def __init__(self, manager):
        """
        iterably load the triples, tokenize and return
        """
        super().__init__()
        self.sequence_length = manager.sequence_length
        self.neighbor_num = manager.neighbor_num

        if manager.rank == 0:
            logger.info(f"initializing DBLP dataset from {self.file_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(manager.plm_dir)

        for k,v in vars(manager).items():
            if k.startswith("enable"):
                setattr(self, k, v)

        self.special_token_ids = set(manager.special_token_ids.values())


    def __len__(self):
        if 'train' in self.file_path:
            return 3009506
        elif "dev" in self.file_path:
            return 60000
        elif "test" in self.file_path:
            return 100000
        else:
            raise NotImplementedError


    def _parse_line(self, line):
        query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
        query_and_neighbors = query_and_neighbors.split('|\'|')[:self.neighbor_num]
        key_and_neighbors = key_and_neighbors.split('|\'|')[:self.neighbor_num]

        query_neighbor_mask = np.zeros(self.neighbor_num, dtype=np.int64)
        query_neighbor_mask[:len(query_and_neighbors)] = 1
        if len(query_and_neighbors) < self.neighbor_num:
            query_and_neighbors += [""] * (self.neighbor_num - len(query_and_neighbors))

        key_neighbor_mask = np.zeros(self.neighbor_num, dtype=np.int64)
        key_neighbor_mask[:len(key_and_neighbors)] = 1
        if len(key_and_neighbors) < self.neighbor_num:
            key_and_neighbors += [""] * (self.neighbor_num - len(key_and_neighbors))

        query_outputs = self.tokenizer(query_and_neighbors, return_tensors="np", padding="max_length", max_length=self.sequence_length, truncation=True)
        query_token_id = query_outputs["input_ids"].astype(np.int64)
        query_attn_mask = query_outputs["attention_mask"].astype(np.int64)

        key_output = self.tokenizer(key_and_neighbors, return_tensors="np", padding="max_length", max_length=self.sequence_length, truncation=True)
        key_token_id = key_output["input_ids"].astype(np.int64)
        key_attn_mask = key_output["attention_mask"].astype(np.int64)

        # default to int64 so that it can be directly converted to long tensor
        return_dict = {
            "query_token_id": query_token_id,
            "key_token_id": key_token_id,
            "query_attn_mask": query_attn_mask,
            "key_attn_mask": key_attn_mask,
            "query_neighbor_mask": query_neighbor_mask,
            "key_neighbor_mask": key_neighbor_mask
        }

        if self.enable_gate == "weight":
            query_gate_mask = np.zeros(query_attn_mask.shape, dtype=np.int64)
            # token_set = set()
            # for i, token_id in enumerate(query_token_id):
            #     for j, token in enumerate(token_id):
            #         if token not in token_set and token not in self.special_token_ids:
            #             query_gate_mask[i, j] = 1
            #             token_set.add(token)
            # token_set = set()
            for i, token_id in enumerate(query_token_id):
                for j, token in enumerate(token_id):
                    if token not in self.special_token_ids:
                        query_gate_mask[i, j] = 1

            key_gate_mask = np.zeros(key_attn_mask.shape, dtype=np.int64)
            # token_set = set()
            # for i, token_id in enumerate(key_token_id):
            #     for j, token in enumerate(token_id):
            #         if token not in token_set and token not in self.special_token_ids:
            #             key_gate_mask[i, j] = 1
            #             token_set.add(token)
            token_set = set()
            for i, token_id in enumerate(key_token_id):
                for j, token in enumerate(token_id):
                    if token not in self.special_token_ids:
                        key_gate_mask[i, j] = 1

            return_dict["query_gate_mask"] = query_gate_mask
            return_dict["key_gate_mask"] = key_gate_mask

        return return_dict


    def __getitem__(self, index):
        line = linecache.getline(self.file_path, index + 1)
        return self._parse_line(line)



class DBLP_Train(DBLP):
    def __init__(self, manager):
        self.file_path = os.path.join(manager.data_root, "DBLP", "train", manager.file_name)
        super().__init__(manager)



class DBLP_Dev(DBLP):
    def __init__(self, manager):
        self.file_path = os.path.join(manager.data_root, "DBLP", "dev", manager.file_name)
        super().__init__(manager)



class DBLP_Test(DBLP):
    def __init__(self, manager):
        self.file_path = os.path.join(manager.data_root, "DBLP", "test", manager.file_name)
        super().__init__(manager)

import re
import os
import sys
import math
import torch
import pickle
import logging
import traceback
import collections
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from random import sample
from queue import Queue
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Callable
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
from torch._six import string_classes

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')




def load_pickle(path):
    """ load pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def download_plm(bert, dir):
    # initialize bert related parameters
    bert_loading_map = {
        "bert": "bert-base-uncased",
        "deberta": "microsoft/deberta-base",
    }
    os.makedirs(dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_loading_map[bert])
    model = AutoModel.from_pretrained(bert_loading_map[bert])
    tokenizer.save_pretrained(dir)
    model.save_pretrained(dir)


def tokenize(sent):
    """ Split sentence into words
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[-\w_]+|[.,!?;|]")

    return [x for x in pat.findall(sent.lower())]


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: mrr scores.
    """
    # descending rank prediction score, get corresponding index of candidate news
    order = np.argsort(y_score)[::-1]
    # get ground truth for these indexes
    y_true = np.take(y_true, order)
    # check whether the prediction news with max score is the one being clicked
    # calculate the inverse of its index
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


class BM25(object):
    """
    compute bm25 score on the entire corpus, instead of the one limited by signal_length
    """
    def __init__(self, k=0.9, b=0.4):
        self.k = k
        self.b = b
        self.logger = logging.getLogger("BM25")


    def fit(self, documents):
        """
        build term frequencies (how many times a term occurs in one news) and document frequencies (how many documents contains a term)
        """
        self.logger.info("Fitting BM25...")
        doc_length = 0
        doc_count = len(documents)

        df = defaultdict(int)
        for document in documents:
            tf = defaultdict(int)
            words = tokenize(document)
            for word in words:
                tf[word] += 1
                df[word] += 1
            doc_length += len(words)

        idf = defaultdict(float)
        for word, freq in df.items():
            idf[word] = math.log((doc_count - freq + 0.5 ) / (freq + 0.5) + 1)

        self.idf = idf
        self.doc_avg_length = doc_length / doc_count


    def __call__(self, document):
        self.logger.info("computing BM25 scores...")

        tf = defaultdict(int)
        words = tokenize(document)
        for word in words:
            tf[word] += 1

        score_pairs = []
        for word, freq in tf.items():
            # skip word such as punctuations
            if len(word) == 1:
                continue
            score = (self.idf[word] * freq * (self.k + 1)) / (freq + self.k * (1 - self.b + self.b * len(document) / self.doc_avg_length))
            score_pairs.append((word, score))
        score_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)
        sorted_document = " ".join([x[0] for x in score_pairs])
        return sorted_document



def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))



class Sequential_Sampler:
    def __init__(self, dataset_length, num_replicas, rank) -> None:
        super().__init__()
        len_per_worker = dataset_length / num_replicas
        self.start = round(len_per_worker * rank)
        self.end = round(len_per_worker * (rank + 1))

    def __iter__(self):
        start = self.start
        end = self.end
        return iter(range(start, end, 1))

    def __len__(self):
        return self.end - self.start



@dataclass
class IterableMultiProcessDataloader:
    dataset: torch.utils.data.IterableDataset
    batch_size: int
    local_rank: int
    world_size: int
    global_end: Any
    collate_fn: Callable = default_collate
    blocking: bool = False
    drop_last: bool = True

    def _start(self):
        self.local_end = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        try:
            batch = []
            for i, sample in enumerate(self.dataset):
                if i % self.world_size != self.local_rank: continue
                batch.append(sample)
                if len(batch) >= self.batch_size:
                    yield self.collate_fn(batch[:self.batch_size])
                    batch = batch[self.batch_size:]
            else:
                if len(batch) > 0 and not self.drop_last:
                    yield self.collate_fn(batch)
                    batch = []
            self.local_end = True
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
            self.pool.shutdown(wait=False)
            raise

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        dist.barrier(device_ids=[self.local_rank])
        while self.aval_count == 0:
            if self.local_end or self.global_end.value:
                self.global_end.value = True
                break
        dist.barrier(device_ids=[self.local_rank])
        if self.global_end.value:
            raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

    def __len__(self):
        return (len(self.dataset) // self.world_size) // self.batch_size



@dataclass
class IterableDataloader:
    dataset: torch.utils.data.IterableDataset
    batch_size: int
    collate_fn: Callable = default_collate
    blocking: bool = False
    drop_last: bool = True

    def _start(self):
        self.local_end = False
        self.aval_count = 0
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        for batch in self._generate_batch():
            self.outputs.put(batch)
            self.aval_count += 1
        self.pool.shutdown(wait=False)
        raise

    def _generate_batch(self):
        batch = []
        for i, sample in enumerate(self.dataset):
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        else:
            if len(batch) > 0 and not self.drop_last:
                yield self.collate_fn(batch)
                batch = []
        self.local_end = True

    def __iter__(self):
        if self.blocking:
            return self._generate_batch()
        self._start()
        return self

    def __next__(self):
        while self.aval_count == 0:
            if self.local_end: raise StopIteration
        next_batch = self.outputs.get()
        self.aval_count -= 1
        return next_batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


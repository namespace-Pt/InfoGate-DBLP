import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-dr", "--data-root", dest="data_root", default="../../../Data")
parser.add_argument("-eg", "--enable-gate", dest="enable_gate", help="way to gate tokens", type=str, choices=["weight", "none", "bm25", "first", "keybert", "random"], default="weight")
parser.add_argument("-nn", "--neighbor-num", dest="neighbor_num", help="number of neighbors", type=int, default=5)

args = parser.parse_args()

text_dir_all = ["train", "dev", "test"]

if args.enable_gate == "bm25":
    from utils.util import BM25
    for text_dir in text_dir_all:
        text_path = os.path.join(args.data_root, "MIND", text_dir, "news.tsv")
        if not os.path.exists(text_path):
            continue

        bm25 = BM25()
        texts = []
        with open(text_path) as f:
            for line in f:
                query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
                query_and_neighbors = query_and_neighbors.split('|\'|')
                key_and_neighbors = key_and_neighbors.split('|\'|')
                texts.extend(query_and_neighbors + key_and_neighbors)

            bm25.fit(texts)

            new_text_path = os.path.join(os.path.split(text_path)[0], "bm25.tsv")
            g = open(new_text_path, "w")
            for line in f:
                query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
                query_and_neighbors = query_and_neighbors.split('|\'|')
                key_and_neighbors = key_and_neighbors.split('|\'|')

                query_and_neighbors = [bm25(x) for x in query_and_neighbors]
                key_and_neighbors = [bm25(x) for x in key_and_neighbors]
                g.write("|\'|".join(query_and_neighbors) + "\t" + "|\'|".join(key_and_neighbors))


elif args.enable_gate == "keybert":
    from keybert import KeyBERT
    kw_model = KeyBERT()

    for text_dir in text_dir_all:
        text_path = os.path.join(args.data_root, "MIND", text_dir, "news.tsv")
        if not os.path.exists(text_path):
            continue

        texts = []
        with open(text_path) as f:
            new_text_path = os.path.join(os.path.split(text_path)[0], "bm25.tsv")
            g = open(new_text_path, "w")
            for line in f:
                query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
                query_and_neighbors = query_and_neighbors.split('|\'|')
                key_and_neighbors = key_and_neighbors.split('|\'|')

                query_and_neighbors = [kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10) for x in query_and_neighbors]
                key_and_neighbors = [kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10) for x in key_and_neighbors]
                g.write("|\'|".join(query_and_neighbors) + "\t" + "|\'|".join(key_and_neighbors))


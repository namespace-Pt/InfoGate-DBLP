import os
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-dr", "--data-root", dest="data_root", default="../../../Data")
parser.add_argument("-eg", "--enable-gate", dest="enable_gate", help="way to gate tokens", type=str, choices=["weight", "none", "bm25", "first", "keybert", "random"], default="weight")

args = parser.parse_args()

text_dir_all = ["train", "dev", "test"]

if args.enable_gate == "bm25":
    from utils.util import BM25
    for text_dir in text_dir_all:
        text_path = os.path.join(args.data_root, "DBLP", text_dir, "base.tsv")
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

        with open(text_path) as f:
            new_text_path = os.path.join(os.path.split(text_path)[0], "bm25.tsv")
            g = open(new_text_path, "w")
            for line in tqdm(f, total=int(subprocess.check_output(["wc", "-l", text_path]).decode("utf-8").split()[0])):
                query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
                query_and_neighbors = query_and_neighbors.split('|\'|')
                key_and_neighbors = key_and_neighbors.split('|\'|')

                query_and_neighbors = [bm25(x) for x in query_and_neighbors]
                key_and_neighbors = [bm25(x) for x in key_and_neighbors]
                g.write("|\'|".join(query_and_neighbors) + "\t" + "|\'|".join(key_and_neighbors) + "\n")
            g.close()


elif args.enable_gate == "keybert":
    from keybert import KeyBERT
    kw_model = KeyBERT()

    for text_dir in text_dir_all:
        text_path = os.path.join(args.data_root, "DBLP", text_dir, "base.tsv")
        if not os.path.exists(text_path) or "train" in text_path:
            continue

        texts = []
        with open(text_path) as f:
            new_text_path = os.path.join(os.path.split(text_path)[0], "keybert.tsv")
            g = open(new_text_path, "w")
            print(text_path)
            for line in tqdm(f, total=int(subprocess.check_output(["wc", "-l", text_path]).decode("utf-8").split()[0])):
                query_and_neighbors, key_and_neighbors = line.strip('\n').split('\t')[:2]
                query_and_neighbors = query_and_neighbors.split('|\'|')[:5]
                key_and_neighbors = key_and_neighbors.split('|\'|')[:5]

                query_and_neighbors_keywords = kw_model.extract_keywords(query_and_neighbors, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
                for i,x in enumerate(query_and_neighbors_keywords):
                    query_and_neighbors_keywords[i] = " ".join([kwd[0] for kwd in x])

                key_and_neighbors_keywords = kw_model.extract_keywords(key_and_neighbors, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
                for i,x in enumerate(key_and_neighbors_keywords):
                    key_and_neighbors_keywords[i] = " ".join([kwd[0] for kwd in x])

                g.write("|\'|".join(query_and_neighbors_keywords) + "\t" + "|\'|".join(key_and_neighbors_keywords) + "\n")
            g.close()


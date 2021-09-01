"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
from sentence_transformers import SentenceTransformer,  util, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datasets import load_dataset
import logging
import sys
import torch
import os
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_base = "https://tfhub.dev/google/"

models = [
    "universal-sentence-encoder-multilingual/3",
    "universal-sentence-encoder-multilingual-large/3",
]

if len(sys.argv) > 1:
    models = sys.argv[1:]

ds = load_dataset('stsb_multi_mt_ja', 'ja', split='test')
ds_en = load_dataset('stsb_multi_mt_ja', 'en', split='test')
    
sentences1 = ds['sentence1']
sentences2 = ds_en['sentence2']
scores = [x/5.0 for x in ds['similarity_score']]

print(sentences1[:3])
print(sentences2[:3])

results = []

class mUSEmodel:
    def __init__(self, model_name):
        self.model = hub.load(model_name)
            
    def encode(self, sentences, batch_size=0, show_progress_bar=None, convert_to_numpy=True):
        res = self.model(sentences)
        return res

    def evaluate(self, evaluator):
        return evaluator(self)

for model_name in models:
    print(model_name)
    model = mUSEmodel(model_base + model_name)
    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, main_similarity=SimilarityFunction.COSINE, name='sts-test')
    spearman_cos = model.evaluate(evaluator)
    results.append('| {:s} | {:.1f} |'.format(model_name, spearman_cos * 100))

print('| model | spearman_cos |')
print('|-------|-------------:|')
for result in results:
    print(result)

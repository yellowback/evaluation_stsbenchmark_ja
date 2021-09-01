"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark_ja.py
OR
python evaluation_stsbenchmark_ja.py model_name
"""
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from datasets import load_dataset
import logging
import sys
import torch
import os

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

models = [
    'sentence-transformers/LaBSE',
    'xlm-roberta-base',
    'xlm-roberta-large',
    'sentence-transformers/quora-distilbert-multilingual',
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/stsb-xlm-r-multilingual',
]

if len(sys.argv) > 1:
    models = sys.argv[1:]

ds = load_dataset('stsb_multi_mt_ja', 'ja', split='test')
ds_en = load_dataset('stsb_multi_mt_ja', 'en', split='test')

sentences1 = ds['sentence1']
sentences2 = ds_en['sentence2']
scores = [x/5.0 for x in ds['similarity_score']]

print(sentences1[:3], sentences2[:3])

results = []

for model_name in models:
    print(model_name)
    model = SentenceTransformer(model_name)
    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, main_similarity=SimilarityFunction.COSINE, name='sts-test')
    spearman_cos = model.evaluate(evaluator)
    results.append('| {:s} | {:.1f} |'.format(model_name, spearman_cos * 100))

print('| model | spearman_cos |')
print('|-------|-------------:|')
for result in results:
    print(result)

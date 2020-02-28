import os
import json
import pickle as pkl
from collections import defaultdict, Counter
from gensim import corpora, models, similarities, matutils

import numpy as np
import pytrec_eval
from tqdm import tqdm

import read_ap
import download_ap
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def print_results(docs, query, n_docs_limit=10, len_limit=50):
    print(f"Query: {query}")
    docs = docs[:n_docs_limit]
    for i, (doc_id, score) in enumerate(docs):
        d = " ".join(docs_by_id[doc_id])
        doc_content = d[:len_limit] + "..."
        print(f"\tRank {i}({score:.2}, {doc_id}): {doc_content}")


class LDA():

    def __init__(self, docs):
        
        # create term document matrices
        matrix_path = "./td_matrices"
        if os.path.exists(matrix_path):
            with open(matrix_path, "rb") as reader:
                matrix = pkl.load(reader)
            self.dictionary = matrix["dictionary"]
            self.bow_corpus = matrix["bow_corpus"]
            self.tfidf_corpus = matrix["tfidf_corpus"]
            self.tfidf_transform = matrix["tfidf_transform"]
      
        else:
            print("Building Dictionary")
            self.dictionary = corpora.Dictionary(list(docs.values()))
            print("Building BOW-matrix")
            self.bow_corpus = [self.dictionary.doc2bow(text) for text in docs.values()]
            print("Building TFIDF-matrix")
            self.tfidf_transform = models.TfidfModel(self.bow_corpus)
            self.tfidf_corpus = self.tfidf_transform[self.bow_corpus]
            
            with open(matrix_path, "wb") as writer:
                matrix = {
                    "dictionary": self.dictionary,
                    "bow_corpus": self.bow_corpus,
                    "tfidf_corpus": self.tfidf_corpus,
                    "tfidf_transform": self.tfidf_transform
                }
                pkl.dump(matrix, writer) 

        # train TFIDF LDA models
        model_path = "./lda_model"
        if os.path.exists(model_path):
            with open(model_path, "rb") as reader:
                model = pkl.load(reader)
            self.bow_model = model["bow_model"]            
            self.tfidf_model = model["tfidf_model"]
            
            
        else:
            print("Training LDA BOW-model")
            self.bow_model = models.LdaModel(self.bow_corpus, id2word=self.dictionary, num_topics=500, dtype=np.float64)
            print("Training LDA TFIDF-model")
            self.tfidf_model = models.LdaModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=500, dtype=np.float64)
            with open(model_path, "wb") as writer:
                model = {
                    "bow_model": self.bow_model,
                    "tfidf_model": self.tfidf_model,
                }
                pkl.dump(model, writer) 
                
   
    def bow_search(self, query, docs):
        query_repr = read_ap.process_text(query)
        query_bow = self.dictionary.doc2bow(query_repr)
        query_lda = self.bow_model[query_bow]
            
        results = defaultdict(float)
        for doc_id, text in docs.items():
            text_bow = self.dictionary.doc2bow(text)
            text_lda = self.bow_model[text_bow]                        
            results[doc_id] = matutils.kullback_leibler(text_lda, query_lda)
                   
        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results   
   
            

if __name__ == "__main__":

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # Create instance for retrieval
    LDA_search = LDA(docs_by_id)
    
    # read in the qrels
    qrels, queries = read_ap.read_qrels()

    print("Running LDA-TFIDF Benchmark")
    overall_ser = {}
    for qid in tqdm(qrels): 
        query_text = queries[qid]

        results = LDA_search.bow_search(query_text, docs_by_id)
        overall_ser[qid] = dict(results)
  
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    with open("LDA-TFIDF.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
   


     

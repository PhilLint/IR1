import os
import json
import pickle as pkl
from collections import defaultdict, Counter
from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
import collections

import numpy as np
import pytrec_eval
from tqdm import tqdm

import read_ap
import download_ap
import logging




def print_results(docs, query, n_docs_limit=10, len_limit=50):
    print(f"Query: {query}")
    docs = docs[:n_docs_limit]
    for i, (doc_id, score) in enumerate(docs):
        d = " ".join(docs_by_id[doc_id])
        doc_content = d[:len_limit] + "..."
        print(f"\tRank {i}({score:.2}, {doc_id}): {doc_content}")


class LSI():

    def __init__(self, texts, n_topics, model_type):
        
        # create term document matrices
        matrix_path = "./lsi_matrices"
        if os.path.exists(matrix_path):
            with open(matrix_path, "rb") as reader:
                matrix = pkl.load(reader)
            self.dictionary = matrix["dictionary"]
            self.bow_corpus = matrix["bow_corpus"]
            self.tfidf_corpus = matrix["tfidf_corpus"]
            self.tfidf_transform = matrix["tfidf_transform"]
      
        else:
            print("Building Dictionary")
            self.dictionary = corpora.Dictionary(texts)
            # filter words occuring in less than 20 docs, or more than 50%
            self.dictionary.filter_extremes(no_below=20, no_above=0.5)
            print("Building BOW-matrix")
            self.bow_corpus = [self.dictionary.doc2bow(text) for text in texts]
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
        
        if model_type == "bow":     
            print("Training LSI BOW-model")
            self.bow_model = models.LsiModel(self.bow_corpus, id2word=self.dictionary, num_topics=n_topics)
            self.bow_index = similarities.MatrixSimilarity(self.bow_model[self.bow_corpus])
        if model_type == "tfidf":  
            print("Training LSI TFIDF-model")
            self.tfidf_model = models.LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=n_topics)
            self.tfidf_index = similarities.MatrixSimilarity(self.tfidf_model[self.tfidf_corpus])
        
        
    def get_model(self, model_type):
        if model_type == "bow":
            return self.bow_model
        elif model_type == "tfidf":
            return self.tfidf_model
            
            
    def get_index(self, model_type):
        if model_type == "bow":
            return self.bow_index
        elif model_type == "tfidf":
            return self.tfidf_index
            
            
    def get_query_repr(self, model_type, query):
        query_repr = read_ap.process_text(query)
        query_vec = self.dictionary.doc2bow(query_repr)       
        if model_type == "tfidf":
            query_vec = self.tfidf_transform[query_vec]
        query_lsi = self.get_model(model_type)[query_vec]
        return query_lsi
        
                
    def search(self, query, doc_ids, model_type):
        query_lsi = self.get_query_repr(model_type, query)
        results = defaultdict(float)        
        sims = self.get_index(model_type)[query_lsi]    
        for i in range(len(sims)):
            results[doc_ids[i]] = float(sims[i])
        
        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results     
                                       
    def benchmark(self, docs, qrels, queries, model_type):
        overall_ser = {}
        print("Running ", model_type, " Benchmark")

        for qid in tqdm(qrels): 
            query_text = queries[qid]
            results = self.search(query_text, list(docs.keys()), model_type)
            overall_ser[qid] = dict(results)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
        metrics = evaluator.evaluate(overall_ser)

        return metrics
 
def trim_text(texts):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    # remove tokens occuring less than 30 times
    texts = [[token for token in text if frequency[token] > 30] for text in texts]
    # remove tokens that are numeric
    texts = [[token for token in doc if not token.isnumeric()] for doc in texts]
    # remove tokens with length = 1
    texts = [[token for token in doc if len(token) > 1] for doc in texts]
    return texts
             
            
def calc_MAP(metrics):
    MAP = 0
    for query in metrics:
           MAP += metrics[query]['map']
    return MAP

def dict_results():
    return collections.defaultdict(dict_results)


def dump_TREC_query(model_type, qid, results):
    if model_type == "bow":
        TREC_path = "./TREC_LSI_BOW.txt"
    if model_type == "tfidf":
        TREC_path = "./TREC_LSI_TFIDF.txt"
    if not os.path.exists(TREC_path):
        with open(TREC_path, "w") as writer:
            pass
    print("Saving Query", qid, "Results")
    with open(TREC_path, "a") as writer:
        for i in range(1000):
            writer.write("{} Q0 {} {} {} STANDARD\n".format(qid, results[i][0], i, results[i][0]))
            
            
def TREC_eval(topic, model_type):
    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    qrels, queries = read_ap.read_qrels()
    
    print("TREC eval")
    test_qrels = dict()
    for k in qrels:
        if (int(k) < 76 or int(k) > 100):
            test_qrels[k] = qrels[k]
             
    model = LSI(docs_by_id, topic, model_type)
    overall_ser = {}
    print("Running ", model_type, " Benchmark")
    for qid in tqdm(test_qrels): 
            query_text = queries[qid]
            results = model.search(query_text, list(docs_by_id.keys()), model_type)
            dump_TREC_query(model_type, qid, results)
    
            
    

def dump_results(BEST_MAP, val_metrics, test_metrics, topic, model_type):
    result_path = "./LSI_results"
    if not os.path.exists(result_path):
        with open(result_path, "wb") as writer:
            results = dict_results()
            pkl.dump(results, writer)
            
    with open(result_path, "rb") as reader:
        results = pkl.load(reader)
    results[model_type]["best"] = BEST_MAP
    results[model_type][topic_n]["val"] = val_metrics
    results[model_type][topic_n]["test"] = test_metrics

    with open(result_path, "wb") as writer:
        pkl.dump(dict(results), writer)
      

def print_top_topics():
    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    qrels, queries = read_ap.read_qrels()
    
    model = LSI(texts, 500, "bow")  
    print("LSI-BOW 5 most significant topics:")
    top_topics = model.bow_model.print_topics(num_topics = 5)
    for topic in top_topics:
        print(topic)
    
    model = LSI(texts, 500, "tfidf")
    print("LSI-TFIDF 5 most significant topics:")
    top_topics = model.tfidf_model.print_topics(num_topics = 5)
    for topic in top_topics:
        print(topic)
    


def tune_params(model_type):
    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    qrels, queries = read_ap.read_qrels()
    
    print("Tuning models")
    val_qrels, test_qrels = dict(), dict()
    for k in qrels:
        if (int(k) > 75 and int(k) < 101):
            val_qrels[k] = qrels[k]
        else:
            test_qrels[k] = qrels[k]
          
    BEST_MAP = 0
    topic_search = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
       
    for topic_n in topic_search:
        print("topic number:", topic_n)
        model = LSI(docs_by_id, topic_n, model_type)
        val_metrics = bow_model.benchmark(docs_by_id, val_qrels, queries, model_type)
        
        MAP = calc_MAP(val_metrics)
        if MAP > BEST_MAP:
            BEST_MAP = MAP
            topic = topic_n            
                    
        test_metrics = model.benchmark(docs_by_id, test_qrels, queries, model_type)
        dump_results(BEST_MAP, val_metrics, test_metrics, topic, model_type)
                    
    return BEST_MAP
    



if __name__ == "__main__":

    print_top_topics()
    best_bow = tune_params("bow")
    best_tfidf = tune_params("tfidf")
    TREC_eval(best_tfidf, "tfidf")
    TREC_eval(best_bow, "bow")
    
    

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



def print_results(docs, query, n_docs_limit=10, len_limit=50):
    print(f"Query: {query}")
    docs = docs[:n_docs_limit]
    for i, (doc_id, score) in enumerate(docs):
        d = " ".join(docs_by_id[doc_id])
        doc_content = d[:len_limit] + "..."
        print(f"\tRank {i}({score:.2}, {doc_id}): {doc_content}")


class LSI():

    def __init__(self, texts, n_topics):
        
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

        # train BOW and TFIDF LSI models         
        print("Training LSI BOW-model")
        self.bow_model = models.LsiModel(self.bow_corpus, id2word=self.dictionary, num_topics=n_topics)
        self.bow_index = similarities.MatrixSimilarity(self.bow_model[self.bow_corpus])
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
    # remove tokens occuring less than 50 times
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


def dump_results(bow_BEST, bow_val, bow_test, tfidf_BEST, tfidf_val, tfidf_test, bow_topic, tfidf_topic):
    result_path = "./LSI_results"
    if not os.path.exists(result_path):
        with open(result_path, "wb") as writer:
            results = dict_results()
            pkl.dump(results, writer)
            
    with open(result_path, "rb") as reader:
        results = pkl.load(reader)
    results["bow"]["best"] = bow_BEST
    results["bow"][topic_n]["val"] = bow_val
    results["bow"][topic_n]["test"] = bow_test
    results["tfidf"]["best"] = tfidf_BEST
    results["tfidf"][topic_n]["val"] = tfidf_val
    results["tfidf"][topic_n]["test"] = tfidf_test
    with open(result_path, "wb") as writer:
        pkl.dump(results, writer)
      



if __name__ == "__main__":
    
    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    qrels, queries = read_ap.read_qrels()
    """
    model = LSI(texts, 5)
     
    print("LSI-BOW 5 most significant topics:")
    top_topics = model.bow_model.print_topics(num_topics = 5)
    for topic in top_topics:
        print(topic)
    
    print("LSI-TFIDF 5 most significant topics:")
    top_topics = model.tfidf_model.print_topics(num_topics = 5)
    for topic in top_topics:
        print(topic)
        
    """
    print("Tuning models")
    val_qrels, test_qrels = dict(), dict()
    for k in qrels:
        if (int(k) > 75 and int(k) < 101):
            val_qrels[k] = qrels[k]
        else:
            test_qrels[k] = qrels[k]
    
    bow_topic, tfidf_topic = 0, 0        
    bow_MAP, tfidf_MAP = 0, 0
    topic_search = [10000]
       
    for topic_n in topic_search:
        print("topic number:", topic_n)
        model = LSI(docs_by_id, topic_n)
        bow_metrics = model.benchmark(docs_by_id, val_qrels, queries, "bow")
        tfidf_metrics = model.benchmark(docs_by_id, val_qrels, queries, "tfidf")
        
        MAP = calc_MAP(bow_metrics)
        if MAP > bow_MAP:
            bow_MAP = MAP
            bow_topics = topic_n            
            
        MAP = calc_MAP(tfidf_metrics)
        if MAP > tfidf_MAP:
            tfidf_MAP = MAP
            tfidf_topics = topic_n
        
        bow_test = model.benchmark(docs_by_id, test_qrels, queries, "bow")
        tfidf_test = model.benchmark(docs_by_id, test_qrels, queries, "tfidf")
        dump_results(bow_MAP, bow_metrics, bow_test, tfidf_MAP, tfidf_metrics, tfidf_test, bow_topics, tfidf_topics)
        
            
    print("Best BOW topic number:", bow_topics)
    print("Best TF-IDF topic number:", tfidf_topics)
    
    with open("./LSI_results", "wb") as writer:
        pkl.dump(all_results, writer) 
    






"""
The vector space representation fails to capture the relationship
between synonymous terms such as car and automobile – according each a
separate dimension in the vector space. Consequently the computed similarity ~q · d~ between a query ~q (say, car) and a document d~ containing both car
and automobile underestimates the true similarity that a user would perceive.
Polysemy on the other hand refers to the case where a term such as charge
has multiple meanings, so that the computed similarity ~q · d~ overestimates
the similarity that a user would perceive. 

Incremental addition fails to capture the co-occurrences of the newly added documents (and even
ignores any new terms they contain). As such, the quality of the LSI representation will degrade as more documents are added and will eventually
require a recomputation of the LSI representation.

When forced to squeeze the terms/documents down to a k-dimensional space, the SVD should bring together terms with similar co-occurrences.

As we reduce k, recall tends to increase, as expected.

a value of k in the low hundreds can actually increase
precision on some query benchmarks. This appears to suggest that for a
suitable value of k, LSI addresses some of the challenges of synonymy.

LSI works best in applications where there is little overlap between queries
and documents.

LSI shares two basic drawbacks of vector
space retrieval: there is no good way of expressing negations (find documents that contain german but not shepherd), and no way of enforcing Boolean
conditions.

LSI can be viewed as soft clustering by interpreting each dimension of the
reduced space as a cluster and the value that a document has on that dimension as its fractional membership in that cluster.
"""

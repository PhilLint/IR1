import os
import json
import pickle as pkl
from collections import defaultdict, Counter
from gensim import corpora, models, similarities

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

        # train BOW and TFIDF LSI models
        model_path = "./lsi_models"
        if os.path.exists(model_path):
            with open(model_path, "rb") as reader:
                model = pkl.load(reader)
            self.bow_model = model["bow_model"]
            self.bow_index = model["bow_index"]
            self.tfidf_model = model["tfidf_model"]
            self.tfidf_index = model["tfidf_index"]
            
        else:
            print("Training LSI BOW-model")
            self.bow_model = models.LsiModel(self.bow_corpus, id2word=self.dictionary, num_topics=500)
            self.bow_index = similarities.MatrixSimilarity(self.bow_model[self.bow_corpus])
            print("Training LSI TFIDF-model")
            self.tfidf_model = models.LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=500)
            self.tfidf_index = similarities.MatrixSimilarity(self.tfidf_model[self.tfidf_corpus])
            with open(model_path, "wb") as writer:
                model = {
                    "bow_model": self.bow_model,
                    "bow_index": self.bow_index,
                    "tfidf_model": self.tfidf_model,
                    "tfidf_index": self.tfidf_index
                }
                pkl.dump(model, writer) 
                
                
    def bow_search(self, query, doc_ids):
        query_repr = read_ap.process_text(query)
        query_bow = self.dictionary.doc2bow(query_repr)
        query_lsi = self.bow_model[query_bow]
        
        results = defaultdict(float)
        sims = self.bow_index[query_lsi]
        for i in range(len(sims)):
            results[doc_ids[i]] = float(sims[i])
        
        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results   
   
    def tfidf_search(self, query, doc_ids):
        query_repr = read_ap.process_text(query)
        query_bow = self.dictionary.doc2bow(query_repr)
        query_tfidf = self.tfidf_transform[query_bow]
        query_lsi = self.tfidf_model[query_tfidf]
        
        results = defaultdict(float)
        sims = self.tfidf_index[query_lsi]
        for i in range(len(sims)):
            results[doc_ids[i]] = float(sims[i])
        
        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results   
            
            

if __name__ == "__main__":

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # Create instance for retrieval
    LSI_search = LSI(docs_by_id)
    
    # read in the qrels
    qrels, queries = read_ap.read_qrels()

    print("Running LSI-BOW Benchmark")
    overall_ser = {}
    for qid in tqdm(qrels): 
        query_text = queries[qid]

        results = LSI_search.bow_search(query_text, list(docs_by_id.keys()))
        overall_ser[qid] = dict(results)
 
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    with open("LSI-BOW.json", "w") as writer:
        json.dump(metrics, writer, indent=1)


    print("Running LSI-TFIDF Benchmark")
    overall_ser = {}
    for qid in tqdm(qrels): 
        query_text = queries[qid]

        results = LSI_search.tfidf_search(query_text, list(docs_by_id.keys()))
        overall_ser[qid] = dict(results)
  
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    with open("LSI-TFIDF.json", "w") as writer:
        json.dump(metrics, writer, indent=1)


    
        
        





































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

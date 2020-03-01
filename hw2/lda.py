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

    def __init__(self, texts):
        
        # create bigrams
        bigram = models.Phrases(texts, min_count=20)
        for idx in range(len(texts)):
            for token in bigram[texts[idx]]:
                if '_' in token:
                    texts[idx].append(token)
        
        
        print("Building Dictionary")
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=20, no_above=0.5)
        print("Building BOW-matrix")
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in texts]
        print("Building TFIDF-matrix")
        self.tfidf_transform = models.TfidfModel(self.bow_corpus)
        self.tfidf_corpus = self.tfidf_transform[self.bow_corpus]


        # train TFIDF LDA models
        model_path = "./lda5_model"
        if os.path.exists(model_path):
            with open(model_path, "rb") as reader:
                model = pkl.load(reader)
            self.bow_model = model["bow_model"]            
            self.tfidf_model = model["tfidf_model"]
            
            
        else:
            print("Training LDA BOW-model")
            self.bow_model = models.LdaModel(self.bow_corpus, id2word=self.dictionary, num_topics=5, dtype=np.float64)
            print("Training LDA TFIDF-model")
            self.tfidf_model = models.LdaModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=5, dtype=np.float64)
            with open(model_path, "wb") as writer:
                model = {
                    "bow_model": self.bow_model,
                    "tfidf_model": self.tfidf_model,
                }
                pkl.dump(model, writer) 
                
   
    def bow_search(self, query, docs):
        query_repr = read_ap.process_text(query)
        query_bow = self.bow_model.id2word.doc2bow(query_repr)
        query_lda = self.bow_model[query_bow]
            
        results = defaultdict(float)
        for doc_id, text in docs.items():
            text_bow = self.dictionary.doc2bow(text)
            text_lda = self.bow_model[text_bow]                        
            results[doc_id] = float(matutils.kullback_leibler(text_lda, query_lda))
                   
        results = list(results.items())
        results.sort(key=lambda _: -_[1])
        return results   
   
            
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


def dump_TREC_query(qid, results):
    TREC_path = "./TREC_LDA.txt"
    
    if not os.path.exists(TREC_path):
        with open(TREC_path, "w") as writer:
            pass
    print("Saving Query", qid, "Results")
    with open(TREC_path, "a") as writer:
        for i in range(1000):
            writer.append("{} Q0 {} {} {} STANDARD".format(qid, results[i][0], i, results[i][0]))
            
            
def TREC_eval():
    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    qrels, queries = read_ap.read_qrels()
    
    print("TREC eval")
    test_qrels = dict()
    for k in qrels:
        if (int(k) < 76 or int(k) > 100):
            test_qrels[k] = qrels[k]
            
    model = LDA(docs_by_id)
    overall_ser = {}
    print("Running Benchmark")
    for qid in tqdm(test_qrels): 
            query_text = queries[qid]
            results = model.search(query_text, list(docs.keys()))
            dump_TREC_query(qid, results)



if __name__ == "__main__":

    docs_by_id = read_ap.get_processed_docs()
    texts = trim_text(list(docs_by_id.values()))
    LDA_search = LDA(texts)
    qrels, queries = read_ap.read_qrels()
    LDA_search.bow_model.print_topics(num_topics = 5)
    
    
    
    TREC_eval()
    
    print("Running LDA-TFIDF Benchmark")
    overall_ser = {}
    for qid in tqdm(qrels): 
        query_text = queries[qid]

        results = LDA_search.bow_search(query_text, docs_by_id)
        overall_ser[qid] = dict(results)
  
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    with open("LDA-BOW.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
        
    
        
    
   


     

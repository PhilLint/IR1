import os
import random
import torch
from torch.optim import SparseAdam
from collections import defaultdict
from data_word2vec import *
from skipgram import SkipGram
from word2vec_utils import *
import read_ap as read_ap
import download_ap as download_ap

import itertools

class word2vec:
    def __init__(self, emb_dimension, min_freq=50, lr=0.001, device="cpu", subset=None):
        """

        Args:
            data_path:
            vocab_size:
            emb_dimension:
            lr:
        """
        self.device = device
        # ensure dataset is downloaded
        download_ap.download_dataset()
        # pre-process the text
        if subset is not None:
            self.docs_by_id = read_ap.get_processed_docs(subset=subset, name_addition=str(subset))
            print("docs by id loaded")
        else:
            self.docs_by_id = read_ap.get_processed_docs()
        self.emb_dimension = emb_dimension
        self.data_indices, self.word_freq, self.word2id, self.id2word = load_dataset(docs=self.docs_by_id, min_freq=min_freq, name_addition=str(subset))
        self.vocabs = list(set(self.data_indices[0]))
        
        print("docs preprocessed")
        print("length data_indices:{}".format(str(len(self.data_indices))))
        self.num_docs = len(self.docs_by_id)
        self.doc_ids = list(self.docs_by_id.keys())
        del self.docs_by_id

        self.vocabulary_size = len(self.data_indices)
        self.model = SkipGram(self.vocabulary_size, emb_dimension)
        self.optim = SparseAdam(self.model.parameters(), lr=lr)


    def train(self, num_epochs=5, window_size=1, batch_size=1024, num_neg_samples=10, save_model_dir='save_model'):
        if not os.path.exists(save_model_dir):
            self.save_model_dir = os.mkdir(save_model_dir)
        else:
            self.save_model_dir = save_model_dir

        self.model = self.model.to(self.device)

        tmp_loss = 0
        dataloader = Dataloader(self.data_indices, self.word_freq ,self.word2id, self.device)

        for step in range(num_epochs):

            pos_u, pos_v, neg_v = dataloader.generate_batch(window_size, batch_size, num_neg_samples)

            self.optim.zero_grad()
            loss = self.model(pos_u, pos_v, neg_v)
            loss.backward()
            self.optim.step()

            tmp_loss += loss.item()

            if step % 100 == 0:
                tmp_loss /= (step+1)
                print('Average loss at step ', step, ': ', tmp_loss)
                tmp_loss = 0

            # save intermediate models in case of errors
            if step % 500 == 0 and step > 0:
                torch.save(self.model.state_dict(), self.save_model_dir + '/model_step_{}.pt'.format(step))

        # save model after done training
        torch.save(self.model.state_dict(),  self.save_model_dir + '/model_step_{}.pt'.format(num_epochs))

    def get_embedding_weights(self):
        weights = self.model.state_dict()
        weight_list = weights['u_emb.weight'].tolist()

        return weight_list

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def most_similar(self, word, top_k=8):
        index = self.word2id[word]
        index = torch.tensor(index, dtype=torch.long).unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.u_emb.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.id2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list

    def get_doc_embeddings(self):
        tmp_mean = []
        for doc in range(self.num_docs):
            tmp_mean.append(aggregate_embeddings(self, doc))
        all_emb = torch.stack(tmp_mean)

        return all_emb




if __name__ == "__main__":

    w2vec = word2vec(200, min_freq=150, lr=0.001, device="cpu", subset=15)
    w2vec.train(window_size=2, num_epochs=5000)

"""
    # read in the qrels
    qrels, queries = read_ap.read_qrels()

    overall_ser = {}
    doc_emb = w2vec.get_doc_embeddings()


    print("Running word2vec Benchmark")
    # collect results
    for qid in qrels:
        print(qid)
        query_text = queries[qid]

        query_repr = read_ap.process_text(query_text)
        query_emb = aggregate_query_emb(w2vec, query_repr)

        if query_emb is None:
            results = {}
            overall_ser[qid] = dict(results)
            continue
            
        query_emb = query_emb.unsqueeze(dim=0)
        
        sims = get_similarities(query_emb, doc_emb)

        ranking, vals = get_ranking(sims, topk=50)

        doc_list, doc_names = get_doc(w2vec, ranking, topk=50)
        results = {}
        for i in range(len(ranking)):
            results[doc_names[i]] = vals[i].item()

        overall_ser[qid] = dict(results)
        """




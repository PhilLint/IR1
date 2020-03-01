import os
import random
import torch
from torch.optim import SparseAdam
from data_word2vec import *
from skipgram import SkipGram

import read_ap as read_ap
import download_ap as download_ap



class word2vec:
    def __init__(self, emb_dimension, min_freq=50, lr=1.0, device="cpu", subset=10):
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
            self.docs_by_id = read_ap.get_processed_docs(subset=1, name_addition="10")
        else:
            self.docs_by_id = read_ap.get_processed_docs()
        print("docs preprocessed")
        self.data_indices, self.word_freq, self.word2id, self.id2word = load_dataset(self.docs_by_id, min_freq)
        print("length data_indices:{}".format(str(len(self.data_indices))))
        print("loaded preprocessed")
        del self.docs_by_id
        self.vocabulary_size = len(set(self.data_indices))
        self.model = SkipGram(self.vocabulary_size, emb_dimension)
        self.optim = SparseAdam(self.model.parameters(), lr=lr)


    def train(self, num_epochs=5, window_size=1, batch_size=128, num_neg_samples=5, output_dir='save_model'):
        if not os.path.exists(output_dir):
            self.outputdir = os.mkdir(output_dir)
        self.model = self.model.to(self.device)

        avg_loss = 0
        dataloader = Dataloader(self.data_indices, self.word_freq ,self.word2id, self.device)

        for step in range(num_epochs):
            pos_u, pos_v, neg_v = dataloader.generate_batch(window_size, batch_size, num_neg_samples)

            loss = self.model(pos_u, pos_v, neg_v)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            avg_loss += loss.item()

            if step % 2000 == 0 and step > 0:
                avg_loss /= 2000
                print('Average loss at step ', step, ': ', avg_loss)
                avg_loss = 0

            # checkpoint
            if step % 100000 == 0 and step > 0:
                torch.save(self.model.state_dict(), self.outputdir + '/model_step%d.pt' % step)

        # save model at last
        torch.save(self.model.state_dict(), self.outputdir + '/model_step%d.pt' % num_epochs**2)

    def save_model(self, out_path):
        torch.save(self.model.state_dict(), out_path + '/model.pt')

    def get_list_vector(self):
        sd = self.model.state_dict()
        return sd['input_emb.weight'].tolist()

    def save_vector_txt(self, path_dir):
        embeddings = self.get_list_vector()
        fo = open(path_dir + '/vector.txt', 'w')
        for idx in range(len(embeddings)):
            word = self.index2word[idx]
            embed = embeddings[idx]
            embed_list = [str(i) for i in embed]
            line_str = ' '.join(embed_list)
            fo.write(word + ' ' + line_str + '\n')
        fo.close()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def vector(self, index):
        self.model.predict(index)

    def most_similar(self, word, top_k=8):
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).cuda().unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.input_emb.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list


if __name__ == "__main__":

    w2vec = word2vec(200, min_freq=50, lr=1.0, device="cpu")
    w2vec.train(window_size=2)
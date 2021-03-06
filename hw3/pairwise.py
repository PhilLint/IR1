import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import stats
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import itertools

import math
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()


class Model():
    def __init__(self, n_feature, n_hidden, learning_rate, sigma):
        self.ranknet = nn.Sequential(
                                nn.Linear(n_feature, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, 1)
                                )
        self.optimizer = optim.SGD(self.ranknet.parameters(), lr=learning_rate)



def eval_model(model, data_fold):
    x = torch.from_numpy(data_fold.feature_matrix).float()
    y = data_fold.label_vector
    model.ranknet.eval()
           
    output = model.ranknet(x)
    output = output.detach().cpu().numpy().squeeze()

    scores = evl.evaluate(data_fold, np.asarray(output))  

    return scores      

    
def train_batch(documentfeatures, labels, model, sig):
    
    model.ranknet.train()
    model.optimizer.zero_grad()
    
    output = model.ranknet(documentfeatures)
    
    loss = pairwiseloss(output, labels, sig)
    
    loss.backward()

    model.optimizer.step()
            
    return model
    
    
def pairwiseloss(preds, labels, sigma):
    preds = preds.squeeze()

    pairs = list(itertools.combinations(range(preds.shape[0]), 2))
    idx1, idx2 = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
   
    S = torch.sign(labels[idx1] - labels[idx2])
    s = preds[idx1] - preds[idx2]   
        
    C = 0.5 * (1 - S) * sigma * s + torch.log(1 + torch.exp(-sigma * s))
 
    return C.sum()   


def hyperparam_search():

    epochs = 30
    learning_rates = [10**-2, 10**-3]
    n_hiddens = [150, 200, 250, 300, 350]
    sigmas = [10**-2, 10**-3]

    best_ndcg = 0
    for learning_rate in learning_rates:
        for n_hidden in n_hiddens:
            for sigma in sigmas:

                print("\nTesting learning_rate = {}, n_hidden = {} and sigma = {}".format(learning_rate, n_hidden, sigma))
                model = Model(data.num_features, n_hidden, learning_rate, sigma)

                last_ndcg = 0
                for epoch in range(epochs):

                    model.ranknet.train()
                    for qid in range(0, data.train.num_queries()):
                        if data.train.query_size(qid) < 2:
                            continue
                        s_i, e_i = data.train.query_range(qid)

                        documentfeatures = torch.tensor(data.train.feature_matrix[s_i:e_i]).float()
                        labels = torch.tensor(data.train.label_vector[s_i:e_i])

                        model = train_batch(documentfeatures, labels, model, sigma)  
              
                    scores = eval_model(model, data.validation)
              
                    ndcg = scores["ndcg"][0]
                    print("Epoch: {}, ndcg: {}".format(epoch, ndcg))
                            
                    if ndcg < last_ndcg:
                        break
                    last_ndcg = ndcg
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_params = {"learning_rate": learning_rate, "n_hidden": n_hidden, "epoch": epoch, "sigma": sigma}            
                        print("Best parameters:", best_params)
    
    return best_params

def train_best(best_params):
    epochs = best_params["epoch"]
    n_hidden = best_params["n_hidden"]
    learning_rate = best_params["learning_rate"]
    sigma = best_params["sigma"]
    
    model = Model(data.num_features, n_hidden, learning_rate, sigma)

    arrs, ndcgs = [], []
    for epoch in range(epochs):
        eval_count = 0
        for qid in range(0, data.train.num_queries()):
            if data.train.query_size(qid) < 2:
                continue
            s_i, e_i = data.train.query_range(qid)
            
            documentfeatures = torch.tensor(data.train.feature_matrix[s_i:e_i]).float()
            labels = torch.tensor(data.train.label_vector[s_i:e_i])
            model = train_batch(documentfeatures, labels, model, sigma) 
            eval_count +=1
            if eval_count % 2000 == 0:
                scores = eval_model(model, data.validation)
                ndcgs.append(scores["ndcg"][0])
                arrs.append(scores["relevant rank"][0])
        print("Epoch: {}, ndcg: {}".format(epoch, scores["ndcg"][0]))
        
    return arrs, ndcgs, model

def plot_ndcg_arr(losses, ndcgs):
    x = np.arange(len(losses))
    fig, ax = plt.subplots()
    
    ax.plot(x, losses, label='ARR')
    ax.plot(x, ndcgs, label='NDCG')
    ax.set_xlabel("Batch % 2000")
    ax.set_ylabel("Score")
    ax.set_title("RankNet LTR")
    legend = ax.legend(loc='upper center')
    
    plt.show()
    

if __name__ == "__main__":
    #determine best hyper parameters
    best_params = hyperparam_search()
    #train best model
    ndcgs, model = train_best(best_params)
    #performance on test set
    scores = eval_model(model, data.test)

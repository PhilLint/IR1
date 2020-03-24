import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import Dataset, TensorDataset, DataLoader
import itertools
import sys

import math
data = dataset.get_dataset().get_data_folds()[0]
data.read_data()


class RankNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(RankNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, 1)      
        
    def forward(self, x1):
        x = torch.nn.functional.relu(self.hidden(x1))
        x = self.output(x)
        
        return x

class Model():
    def __init__(self, n_feature, n_hidden, learning_rate, sigma):
        self.ranknet = RankNet(n_feature, n_hidden)
        self.optimizer = torch.optim.SGD(self.ranknet.parameters(), lr=learning_rate)


def eval_model(model, data_fold):
    x = torch.from_numpy(data_fold.feature_matrix).float()
    y = data_fold.label_vector
    model.ranknet.eval()
           
    output = model.ranknet(x)
    output = output.detach().cpu().numpy().squeeze()

    scores = evl.evaluate(data_fold, np.asarray(output))  

    return scores      

    
def train_batch(documentfeatures, labels, model, sig, IRM):
    
    model.ranknet.train()
    model.optimizer.zero_grad()
    
    output = model.ranknet(documentfeatures)
    
    loss = listwiseloss(output, labels, sig, IRM)
    
    loss.sum().backward()

    model.optimizer.step()
            
    return model


    
def listwiseloss(preds, labels, sigma, IRM="ndcg"):
    
    preds = preds.squeeze()

    pairs = list(itertools.combinations(range(preds.shape[0]), 2))
    idx1, idx2 = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
   
    S = torch.sign(labels[idx1] - labels[idx2])
    s = preds[idx1] - preds[idx2] 

    lambda_ij = sigma * (0.5 * (1 - S) - (1 / (1 + torch.exp(sigma * s))))
        
    sort_ind = np.argsort(preds.detach().numpy())[::-1]
    sorted_labels = labels.numpy()[sort_ind]
    ideal_labels = np.sort(labels)[::-1]
    
    sorted_M = np.tile(sorted_labels, (lambda_ij.shape[0], 1))
    for i, pair in enumerate(pairs):
        sorted_M[i][sort_ind[pair[0]]], sorted_M[i][sort_ind[pair[1]]] = sorted_M[i][sort_ind[pair[1]]], sorted_M[i][sort_ind[pair[0]]]   
    sorted_M = torch.from_numpy(sorted_M)
    
    if IRM == "ndcg":
        deltaIRM = get_delta_ndcg(sorted_labels, ideal_labels, sorted_M)
    else:
        deltaIRM = get_delta_ERR(sorted_labels, ideal_labels, sorted_M)

    C = lambda_ij * deltaIRM

    lambda_i = torch.zeros((labels.shape[0], labels.shape[0]))
    lambda_i[np.triu_indices(labels.shape[0], k=1)] = C.float()
    lambda_i = (lambda_i - lambda_i.T).sum(1)
    
    return preds * lambda_i.detach()




def get_delta_ERR(sorted_labels, ideal_labels, sorted_M):
  
    ideal_ERR = evl.sorting_ERR(ideal_labels)
    n_ERR = evl.ERR(sorted_labels, ideal_labels)
    
    ERRs = []
    r_i = (torch.pow(2, sorted_M) - 1) / (2**4)
    for row in r_i:
        ERR, t = 0, 1
        for m in range(1, len(row)):
            ERR += t*row[m]/m
            t *= (1-row[m])
        if ERR == 0:
            ERR = 1
        ERRs.append(ERR)
    deltaIRM = np.abs((np.array(ERRs) / np.array(ideal_ERR)) - n_ERR)
    return torch.tensor(deltaIRM)




def get_delta_ndcg(sorted_labels, ideal_labels, sorted_M):
    
    dcg = evl.dcg_at_k(sorted_labels, 0)
    idcg = evl.dcg_at_k(ideal_labels, 0)
    ndcg = dcg/idcg
    
    denom = torch.tensor(1./np.log2(np.arange(sorted_labels.shape[0])+2.))
    nom = torch.pow(sorted_M, 2) - 1.
    ndcgs = (nom * denom).sum(1) / idcg

    return torch.abs(ndcgs - ndcg)

def get_ranked_labels(scores, labels):
    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind]
    ideal_labels = np.sort(labels)[::-1]

    return sorted_labels, ideal_labels


def calc_dcg(labels):
    k = labels.shape[0]
    denom = 1./np.log2(np.arange(k)+2.)
    nom = 2**labels-1.
    return np.sum(nom[:k]*denom)
    
  
def hyperparam_search():

    epochs = 13
    learning_rates = [10**-2, 10**-3]
    n_hiddens = [150, 200, 250, 300, 350]
    sigmas = [10**-2, 10**-3]
    IRMs = ["ndcg", "ERR"]

    best_ndcg = 0
    for learning_rate in learning_rates:
        for n_hidden in n_hiddens:
            for sigma in sigmas:
                for IRM in IRMs:

                    print("\nTesting learning_rate = {}, n_hidden = {}, IRM = {} and sigma = {}".format(learning_rate, n_hidden, IRM, sigma))
                    model = Model(data.num_features, n_hidden, learning_rate, sigma)

                    last_ndcg = 0
                    for epoch in range(epochs):

                        model.ranknet.train()
                        for qid in range(0, data.train.num_queries()):
                            if data.train.query_size(qid) < 2 or not any(data.train.query_labels(qid) > 0):
                                continue
                            s_i, e_i = data.train.query_range(qid)
                            



                            documentfeatures = torch.tensor(data.train.feature_matrix[s_i:e_i]).float()
                            labels = torch.tensor(data.train.label_vector[s_i:e_i])

                            model = train_batch(documentfeatures, labels, model, sigma, IRM)  
                  
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

    losses, ndcgs = [], []
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
        print("Epoch: {}, ndcg: {}".format(epoch, scores["ndcg"][0]))
        
    return ndcgs, model
    

if __name__ == "__main__":
    #determine best hyper parameters
    best_params = hyperparam_search()
    #train best model
    ndcgs, model = train_best(best_params)
    #performance on test set
    scores = eval_model(model, data.test)

            
      

    


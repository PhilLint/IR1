import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools

from scipy import stats
from torch.utils.data import Dataset, TensorDataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

data = dataset.get_dataset().get_data_folds()[0]
data.read_data()


class Model():
    def __init__(self, n_feature, n_hidden, learning_rate):
        self.net = nn.Sequential(
                                 nn.Linear(n_feature, n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, 1)
                                ).to(device)
        self.criterion = pair_loss
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)


def pair_loss(output, y, sigma):
    output = output.squeeze()
    
    if y.shape[0] == 1:
        return (output - y)**2

    pairs = list(itertools.combinations(range(output.shape[0]), 2))
    val1, val2 = [x[0] for x in pairs], [x[1] for x in pairs]

    pred1 = output[val1].to(device)
    pred2 = output[val2].to(device)
    
    true1 = y[val1]
    true2 = y[val2]

    l1 = (true1 > true2).type(torch.ByteTensor).to(device)
    l2 = (true1 < true2).type(torch.ByteTensor).to(device)
    
    S =  l1 - l2
    
    #lambda_ij = sigma*(torch.FloatTensor([0.5])*(torch.FloatTensor([1])-S)-(torch.FloatTensor([1])/(torch.FloatTensor([1])+torch.exp((pred1-pred2)*sigma))))
    lambda_ij = sigma * (0.5 * (1 -S) - (1 / 1 + torch.exp((pred1 - pred2) * sigma)))
    
    lambda_i = np.zeros(len(set(val1))+1)
    n=0
    for i in val1:
        lambda_i[i]+= lambda_ij[n]
        n += 1
    
    m = 0
    for j in val2:
        lambda_i[j]-= lambda_ij[m] 
        m += 1
    lambda_i = torch.tensor(lambda_i, requires_grad=True)
    
    return predictedvals * lambda_i.detach()    
    

def eval_model(model, data_fold):
    with torch.no_grad():
        x = torch.from_numpy(data_fold.feature_matrix).float().to(device)
        y = data_fold.label_vector
        model.net.eval()
               
        output = model.net(x)      
          
        output = output.detach().cpu().numpy().squeeze()
        
        scores = evl.evaluate(data_fold, np.asarray(output))  

    return scores

def calc_ERR(model, data_fold):
    ERR = 0
    for qid in range(data_fold.num_queries()):
                    
        s_i, e_i = data.train.query_range(qid)
        x = torch.from_numpy(data.train.feature_matrix[s_i:e_i]).float()
        y = torch.from_numpy(data.train.label_vector[s_i:e_i])
        
        output = model.net(x)
        
        

    
def train_batch(x_batch, y_batch, model, sigma):
    model.net.train()
    model.optimizer.zero_grad()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device) 
           
    output = model.net(x_batch)
    loss = model.criterion(output, y_batch, sigma)
    
    loss.backward()
    model.optimizer.step()
    
    return model

       
def hyperparam_search():
    # hyper-parameters
    epochs = 2
    learning_rates = [10**-1, 10**-2, 10**-3, 10**-4]
    n_hiddens = [100, 150, 200, 250, 300, 350, 400]
    sigmas = [0.1, 1, 10, 100]
    
    best_ndcg = 0
    for learning_rate in learning_rates:
        for n_hidden in n_hiddens:
            for sigma in sigmas:
        
                print("\nTesting learning_rate = {}, n_hidden = {} and sigma = {}".format(learning_rate, n_hidden, sigma))
                model = Model(data.num_features, n_hidden, learning_rate)
                switch = False
                last_ndcg = 0
                for epoch in range(epochs):
                    
                    model.net.train()
                    for qid in range(data.train.num_queries()):
                        
                        s_i, e_i = data.train.query_range(qid)
                        x_batch = torch.from_numpy(data.train.feature_matrix[s_i:e_i]).float()
                        y_batch = torch.from_numpy(data.train.label_vector[s_i:e_i])
                        
                        model = train_batch(x_batch, y_batch, model, sigma)  
                    
                    scores = eval_model(model, data.validation)                                    
                    ndcg = scores["ndcg"][0]
                                
                    if ndcg < last_ndcg:
                        if switch: break
                        switch = True
                                   
                    last_ndcg = ndcg
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_params = {"learning_rate": learning_rate, "n_hidden": n_hidden, "epoch": epoch, "sigma": sigma}            
                        print("Best parameters:", best_params)
                    print("Epoch: {}, ndcg: {}".format(epoch, ndcg))
    
    return best_params
    
    
def train_best(best_params):
    epochs = best_params["epoch"]
    n_hidden = best_params["n_hidden"]
    learning_rate = best_params["learning_rate"]
    sigma = best_params["sigma"]
    
    model = Model(data.num_features, n_hidden, learning_rate)

    ndcgs = []
    for epoch in range(epochs):
        eval_count = 0
        for qid in range(data.train.num_queries()):
                    
            s_i, e_i = data.train.query_range(qid)
            x_batch = torch.from_numpy(data.train.feature_matrix[s_i:e_i]).float()
            y_batch = torch.from_numpy(data.train.label_vector[s_i:e_i])

            model = train_batch(x_batch, y_batch, model, sigma)
            
            eval_count +=1
            if eval_count % 100 == 0:
                scores = eval_model(model, data.validation)
                ndcgs.append(scores["ndcg"][0])
        
        print("Epoch: {}, ndcg: {}".format(epoch, scores["ndcg"][0]))
        
    return ndcgs, model
    

if __name__ == "__main__":
    #determine best hyper parameters
    best_params = hyperparam_search()
    #train best model
    ndcgs, losses, model = train_best(best_params)
    #performance on test set
    scores = eval_model(model, data.test)
    error = calc_err(model, data.test)

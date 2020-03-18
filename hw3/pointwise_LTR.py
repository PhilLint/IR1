import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import Dataset, TensorDataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, 1)      
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.output(x)
        return x

class Model():
    def __init__(self, n_feature, n_hidden, learning_rate):
        self.net = Net(n_feature, n_hidden).to(device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)


def eval_model(model, data_fold):
    with torch.no_grad():
        x = torch.from_numpy(data_fold.feature_matrix).float().to(device)
        y = data_fold.label_vector
        model.net.eval()
               
        output = model.net(x)
        output = output.detach().cpu().numpy().squeeze()
        
        loss = np.mean(np.square(output - y))
        scores = evl.evaluate(data_fold, np.asarray(output))  

    return loss, scores


def load_dataset():
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()

    train_x = torch.from_numpy(data.train.feature_matrix).float()
    train_y = torch.from_numpy(data.train.label_vector).float()

    train_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

    return data, train_loader
       

def plot_ndcg_loss(losses, ndcgs):
    x = np.arange(len(losses))
    fig, ax = plt.subplots()
    
    ax.plot(x, losses, label='Loss')
    ax.plot(x, ndcgs, label='NDCG')
    ax.set_xlabel("Batch % 2000")
    ax.set_ylabel("Score")
    ax.set_title("Pointwise LTR")
    legend = ax.legend(loc='upper center')
    
    plt.show()
    plt.savefig('Pointwise_LTR_plot.png')

    
def train_batch(x_batch, y_batch, model):
    model.net.train()
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
           
    output = model.net(x_batch)
    if output.size() != y_batch.size():
        y_batch = y_batch.view(-1, 1)
    loss = model.criterion(y_batch, output)
    
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad() 
    
    return model
    
       
def hyperparam_search():
    # hyper-parameters
    epochs = 300
    learning_rates = [10**-1, 10**-2, 10**-3, 10**-4]
    n_hiddens = [100, 150, 200, 250, 300, 350, 400]
    data, train_loader = load_dataset()
    
    best_ndcg = 0
    for learning_rate in learning_rates:
        for n_hidden in n_hiddens:
        
            print("\nTesting learning_rate = {} and n_hidden = {}".format(learning_rate, n_hidden))
            model = Model(data.num_features, n_hidden, learning_rate)
            
            last_ndcg = 0
            for epoch in range(epochs):
                
                model.net.train()
                for x_batch, y_batch in train_loader:
                    model = train_batch(x_batch, y_batch, model)                          
                loss, scores = eval_model(model, data.validation)
                
                ndcg = scores["ndcg"][0]
                print("Epoch: {}, ndcg: {}".format(epoch, ndcg))
                            
                if ndcg < last_ndcg:
                    break
                last_ndcg = ndcg
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_params = {"learning_rate": learning_rate, "n_hidden": n_hidden, "epoch": epoch}            
                    print("Best parameters:", best_params)
    
    return best_params
    
    
def train_best(best_params):
    epochs = best_params["epoch"]
    n_hidden = best_params["n_hidden"]
    learning_rate = best_params["learning_rate"]
    
    #load data
    data, train_loader = load_dataset()
    model = Model(data.num_features, n_hidden, learning_rate)

    losses, ndcgs = [], []
    for epoch in range(epochs):
        eval_count = 0
        for x_batch, y_batch in train_loader:
            model = train_batch(x_batch, y_batch, model)
            eval_count +=1
            if eval_count % 2000 == 0:
                loss, scores = eval_model(model, data.validation)
                losses.append(loss)
                ndcgs.append(scores["ndcg"][0])
        print("Epoch: {}, ndcg: {}".format(epoch, scores["ndcg"][0]))
        
    return ndcgs, losses, model


def get_distributions(model):
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    model.net.eval()

    val_x = torch.from_numpy(data.validation.feature_matrix).float().to(device)
    test_x = torch.from_numpy(data.test.feature_matrix).float().to(device)
           
    val = model.net(val_x).detach().cpu().numpy().squeeze()
    test = model.net(test_x).detach().cpu().numpy().squeeze()
    actual = np.concatenate((data.train.label_vector, data.validation.label_vector, data.test.label_vector))
    
    distributions = {
    "val_mean": np.mean(val),
    "val_std": np.std(val),
    "test_mean": np.mean(test),
    "test_std": np.std(test),
    "actual_mean": np.mean(actual),
    "actual_std": np.std(actual),
    }
    
    return distributions



if __name__ == "__main__":
    #determine best hyper parameters
    best_params = hyperparam_search()
    #train best model
    ndcgs, losses, model = train_best(best_params)
    #plot ndcg and loss    
    plot_ndcg_loss(losses, ndcgs)
    #get distributions of scores
    distributions = get_distributions(model)
    #performance on test set
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    loss, scores = eval_model(model, data.test)


        
        
















































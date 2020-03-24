import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
import copy

from scipy import stats
from torch.utils.data import Dataset, TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

data = dataset.get_dataset().get_data_folds()[0]
data.read_data()


class RankNetSU(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(RankNetSU, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, 1)

    def forward(self, x1):
        x = torch.nn.functional.relu(self.hidden(x1))
        x = self.output(x)

        return x


class Model():
    def __init__(self, n_feature, n_hidden, learning_rate):
        self.ranknet = RankNetSU(n_feature, n_hidden)
        # .to(device)
        self.optimizer = torch.optim.SGD(self.ranknet.parameters(), lr=learning_rate)


def eval_model(model, data_fold):
    # with torch.no_grad():
    x = torch.from_numpy(data_fold.feature_matrix).float()
    y = data_fold.label_vector
    model.ranknet.eval()

    output = model.ranknet(x)
    output = output.detach().cpu().numpy().squeeze()

    scores = evl.evaluate(data_fold, np.asarray(output))

    return scores


def load_dataset():
    train_x = torch.from_numpy(data.train.feature_matrix).float()
    train_y = torch.from_numpy(data.train.label_vector).float()

    train_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

    return train_loader


def plot_ndcg_loss(losses, ndcgs):
    x = np.arange(len(losses))
    fig, ax = plt.subplots()

    ax.plot(x, losses, label='Loss')
    ax.plot(x, ndcgs, label='NDCG')
    ax.set_xlabel("Batch % 2000")
    ax.set_ylabel("Score")
    ax.set_title("LambdaRank")
    legend = ax.legend(loc='upper center')

    plt.show()
    plt.savefig('LambdaRank_plot.png')


def train_batch(documentfeatures, labels, model, sig):
    #     model.ranknet.train()
    for epoch in range(1):
        #         for qid in range(0, train_data.num_queries()):
        #             if train_data.query_size(qid) < 2:
        #                 continue
        model.optimizer.zero_grad()

        output = model.ranknet(documentfeatures)

        loss = listwiseloss(output, labels, sig)

        # AttributeError: 'Tensor' object has no attribute 'forward'

        loss.sum().backward()

        model.optimizer.step()

    return model

def get_delta_ndcg_ij(output, labels):
    with torch.no_grad():
        scores = output.detach().cpu().numpy().squeeze()
        n_docs = scores.shape[0]
        labels = labels.numpy()

        sorted_labels, ideal_labels = get_ranked_labels(scores, labels)
        # ndcg of original ranking withpout swapping
        #print("sorted labels: {}".format(sorted_labels))
        #print("ideal labels: {}".format(ideal_labels))
        idcg = evl.dcg_at_k(ideal_labels, k=0)

        if idcg == 0:
            #print("idcg {}".format(idcg))
            #print("output {}".format(output))
            #print("labels {}".format(labels))

            delta_ndcg_ij = np.ones((n_docs, n_docs))

            return delta_ndcg_ij[np.triu_indices(n_docs, k=1)]

        ndcg = evl.dcg_at_k(sorted_labels, k=0) / idcg
        #print("ndcg reference: {}".format(ndcg))

        # tensor of shape (n_docs, n_docs) to be multiplied to lambda_ij
        delta_ndcg_ij = np.ones((n_docs, n_docs))

        for i in range(n_docs):
            for j in range(n_docs):
                # copy scores and labels to be swapped for delta irm calculation
                scores_tmp = np.copy(scores)
                # swap scores of i and j to be then changed in the respective ranking
                scores_tmp[i], scores_tmp[j] = scores_tmp[j], scores_tmp[i]
                # new sorted labels after swapping
                sorted_labels_tmp, _ = get_ranked_labels(scores_tmp, labels)
                # new ndcg measure after swapping
                ndcg_tmp = evl.dcg_at_k(sorted_labels_tmp, k=0)/ idcg
                # print("ndcg swap: {}".format(ndcg_tmp))
                delta_ndcg_ij[i, j] = np.abs(ndcg_tmp - ndcg)
                # print("delta_ndcg_ij {}".format(delta_ndcg_ij))

        return delta_ndcg_ij[np.triu_indices(n_docs, k=1)]

def get_ranked_labels(scores, labels):
    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind]
    ideal_labels = np.sort(labels)[::-1]

    return sorted_labels, ideal_labels


def listwiseloss(preds, labels, sigma, irm_type="ndcg"):
    preds = preds.squeeze()

    np.set_printoptions(linewidth=250)

    pairs = list(itertools.combinations(range(preds.shape[0]), 2))
    idx1, idx2 = [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    S = torch.sign(labels[idx1] - labels[idx2])
    s = preds[idx1] - preds[idx2]

    lambda_ij = sigma * (0.5 * (1 - S) - (1 / (1 + torch.exp(sigma * s))))

    lambda_i = torch.FloatTensor(torch.zeros((preds.shape[0], preds.shape[0])))

    if irm_type == "ndcg":
        delta_irm_ij  = get_delta_ndcg_ij(preds, labels)
    #else:
    #    delta_irm_ij  = get_delta_err_ij(preds, labels)

    lambda_i[np.triu_indices(preds.shape[0], k=1)] = lambda_ij * torch.FloatTensor(delta_irm_ij)
    lambda_i = (lambda_i - lambda_i.T).sum(1)

    return preds * lambda_i.detach()


def hyperparam_search():
    # hyper-parameters
    # epochs = 10
    # learning_rates = [10**-1, 10**-2, 10**-3, 10**-4]
    # n_hiddens = [100, 150, 200, 250, 300, 350, 400]
    # sigma = [0.1, 1, 10, 60, 100]
    epochs = 10
    learning_rates = [10 ** -2]#, 10 ** -3, 10 ** -1]
    n_hiddens = [150]#, 200, 250, 300, 350]
    sigmas = [10 ** -2]#, 10 ** -3]

    best_ndcg = 0
    for learning_rate in learning_rates:
        for n_hidden in n_hiddens:
            for sig in sigmas:

                print("\nTesting learning_rate = {}, n_hidden = {} and sigma = {}".format(learning_rate, n_hidden, sig))
                model = Model(data.num_features, n_hidden, learning_rate)

                last_ndcg = 0
                for epoch in range(epochs):

                    model.ranknet.train()
                    qid_list = list(range(0, data.train.num_queries()))
                    # shuffle qids
                    np.random.shuffle(qid_list)
                    cnt = 0
                    for qid in qid_list:
                        cnt += 1
                        if cnt % 1000 == 0:
                            print("{} queries processed".format(cnt))

                        if data.train.query_size(qid) < 2 or not any(data.train.query_labels(qid) > 0):
                            # print("qid {} with labels being zero or only one doc".format(qid))
                            continue

                        s_i, e_i = data.train.query_range(qid)

                        documentfeatures = torch.tensor(data.train.feature_matrix[s_i:e_i]).float()
                        labels = torch.tensor(data.train.label_vector[s_i:e_i])

                        if documentfeatures.shape[0] > 30:
                            documentfeatures = documentfeatures[:30,:]
                            labels = labels[:30]
                            #print("doc shape {}".format(documentfeatures.shape[0]))
                            #print("labels shape {}".format(labels.shape[0]))

                        model = train_batch(documentfeatures, labels, model, sig)

                        if cnt % 5000 == 0:
                            scores = eval_model(model, data.validation)

                            ndcg = scores["ndcg"][0]
                            print("Epoch: {}, query: {}, ndcg: {}".format(epoch, cnt, ndcg))

                    scores = eval_model(model, data.validation)

                    ndcg = scores["ndcg"][0]
                    print("Epoch: {}, ndcg: {}".format(epoch, ndcg))

                    if ndcg < last_ndcg:
                        break
                    last_ndcg = ndcg
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_params = {"learning_rate": learning_rate, "n_hidden": n_hidden, "epoch": epoch,
                                       "sigma": sig}
                        print("Best parameters:", best_params)

    return best_params

# def get_topk_docs_per_query(scores, labels):
#
#
#
#
#
#
#

def train_best(best_params):
    epochs = best_params["epoch"]
    n_hidden = best_params["n_hidden"]
    learning_rate = best_params["learning_rate"]
    sigma = best_params["sigma"]

    model = Model(data.num_features, n_hidden, learning_rate, sigma)

    losses, ndcgs = [], []
    for epoch in range(epochs):
        eval_count = 0
        for qid in range(0, data.train.num_queries()):  #
            if data.train.query_size(qid) < 2:  #
                continue  #
            s_i, e_i = data.train.query_range(qid)

            documentfeatures = torch.tensor(data.train.feature_matrix[s_i:e_i]).float()
            labels = torch.tensor(data.train.label_vector[s_i:e_i])

            model = train_batch(documentfeatures, labels, model, sigma)
            eval_count += 1
            if eval_count % 2000 == 0:
                scores = eval_model(model, data.validation)
                ndcgs.append(scores["ndcg"][0])
        print("Epoch: {}, ndcg: {}".format(epoch, scores["ndcg"][0]))

    return ndcgs, model


def get_distributions(model):
    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()
    model.ranknet.eval()

    val_x = torch.from_numpy(data.validation.feature_matrix).float()
    test_x = torch.from_numpy(data.test.feature_matrix).float()

    val = model.ranknet(val_x).detach().cpu().numpy().squeeze()
    test = model.ranknet(test_x).detach().cpu().numpy().squeeze()
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
    # determine best hyper parameters
    best_params = hyperparam_search()
    # train best model
    #ndcgs, model = train_best(best_params)
    #print("ndcgs {}:".format(ndcgs))
    # plot ndcg and loss
    #plot_ndcg_loss(ndcgs)
    # get distributions of scores
    #distributions = get_distributions(model)
    # performance on test set
    #data = dataset.get_dataset().get_data_folds()[0]
    #data.read_data()
    #scores = eval_model(model, data.test)

# Assignment 3

## Pointwise Learning to Rank 

We implemented the pointwise LTR approach both as a classification and regression task. However, for the final notebook and the 
report, we focused on the regression task. Both approaches can be performed by calling: 

```
pointwise_class.py
pointwise_reg.py
```

More precisely, each script performs a hyperparameter search for the following learning rates, number of hidden layers
and fixed number of epochs / batch_size. Early stopping was assessed based on the change in validation ndcg performance. 
In case the current ndcg is worse than the one of the previous epoch, an early stop is initiated. We also include 
a run without early stopping for the pointwise LTR in the PointwiseLTR notebook to demonstrate overtraining after
the early stopping. Thus, our early stopping criterion is justified. 

```
epochs = 300
batch_size 32
learning_rates = [10**-1, 10**-2, 10**-3, 10**-4]
n_hiddens = [100, 150, 200, 250, 300, 350, 400]
```

After that, the best parametersetting is used to train a 'best' model, for which a loss/performance figure is plotted. 
Then, the final trained model is evaluated on the test data. We provide these experiments in the form of a notebook, as 
well, which states the parameters mentioned in the report.  

```
HW3_output.ipynb
```

### Rubric 
We want to briefly link the given points in the rubric section of the assignment sheet to respective functions/parts in 
our codebase. The following functions are all found within *pointwise_reg.py*.

* Training a Pointwise model: *train_batch(x_batch, y_batch, model)* trains a given model for a batch of documents and 
respective labels. This function does the actual model training with parameter changes, whereas *train_best()* and 
*hyperparam_search()* draw on the basic train function. The former trains the model for the best parameter setting, which 
was obtained by the hyperparameter search.

* Function to evaluate the Pointwise model: Given a trained model and a certain split of the dataset, 
*eval_model(model, data_fold)* evaluates the pointwise model. Computed performance measures are found in *evaluate.py*

* Hyperparameter search and early stopping using the validation set: *hyperparam_search()* does all of this and returns 
a dict with the best parameter setting based on the available parameter values. 

The remaining points are found in the Report. 

## Pairwise Learning to Rank: RankNet and spedup RankNet

Due to the slower training for both pairwise and listwise approaches, we trimmed the hyperparameter search to the following 
values:

```
epochs = 13
learning_rates = [10**-2, 10**-3]
n_hiddens = [150, 200, 250, 300, 350]
sigmas = [10**-2, 10**-3]
```

The ordinary RankNet with the slower lossfunction can be computed by running: 

```
pairwise.py
```

It follows the exact same basic structure as the pointwise LTR script. However, minor changes, such as the new pairwise 
loss function, an additional hyperparameter sigma and the ERR performance measure on the test split appear.\

The spedup version of part 3.b of the assignment can be run by calling 

```
pairwiseSU.py
```

This script only differs in the computation of the loss function. The results of our experiments can moreover be found 
in lower part of the PointwiseLTR notebook. 

### Rubric 
Again, the mentioned points are briefly linked to parts in the code. 

* RankNet module class with hyperparameters: *class RankNet(torch.nn.Module)* within the *pairwise.py* files with 
initialization of the hyperparameters and a forward method for a forward pass of the shallow neural network. 

* The train, the eval and the hyperparameter search functions are very similar to the respective functions of *pointwise.py*

Optimal parameters according to the hyperparameter search are reported and found in the output notebook. 

* Implementation of Sped-up RankNet: As mentioned above, the spedup RankNet can be found in *pairwiseSU.py*. It differs 
to the non spedup version only in the calculation of the pairwise loss. More precisely, *pairwiseloss(preds, labels, sigma)*
now includes the lambda_ij calculation as mentioned in the assignment sheet. The rest is copied from the ordinary
RankNet. If we had had more time, we could have merged the two files into one which then only used a different loss 
function based on the version of RankNet that ought to be computed. That would save redundant code.  

## Listwise Learning to Rank: LambdaRank 

LambdaRank is very similar to the spedup version of RankNet, which is why we used the same basic structure of the 
pairwiseSU.py file. However, as it is listwise, list based measures need to be performed within the training loss 
calculation. At first, we solved this with for-loops, which can be found in:

```
listwise_noop_for_loops.py
```

As all of our files follow the same basic structure, running the scripts will yield the same as for the point- and pairwise
approaches. Due to time constraints, the hyperparameter search is trimmed and analogously to the pairwise approach, 
additionally to the number of hidden layers and learning rate, the sigma parameter is a hyperparameter. Furthermore, 
the IRM measure being NDCG or ERR is an additional hyperparameter.

Though this script works, the slow training led us to come up with a vectorized version of the LambdaRank with NDCG, which 
can be computed with:

```
listwise.py
```

The scripts were run and are included in 

```
HW3_output.ipynb
```

### Rubric 

* Implementation of delta ERR: *get_delta_ERR(sorted_labels, ideal_labels, sorted_M)* calculates the delta ERR for each 
document based on the ranked labels according to the scores of the model, the ideal ranking based on the actual relevanc 
scores and the different combinations of rankings in tensor form. 

* Implementation of delta NDCG: Analogously, *get_delta_ndcg(sorted_labels, ideal_labels, sorted_M)* does the same for 
ndcg as IRM measure. 

* Implementation of LambdaRank (IR Measure should be a hyperparameter): We reuse the classes written for RankNet with the 
difference in loss function, which now is calculated by *listwiseloss(preds, labels, sigma, IRM="ndcg")* allowing 
for the IRM type to be changed as a hyperparameter. 

* The remaining functions, such as hyperparameter search, training and evaluation is analogous to the functions of the 
pointwise LTR. A small adjustment has been made in the training. As some queries only contain documents with label zero, 
the corresponding ndcg and err would involve dividing by zero. Therefore, we skipped those queries, same as for queries 
containing only one document. 

 



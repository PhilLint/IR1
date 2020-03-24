# Assignment 3

## Pointwise Learning to Rank 

We implemented the pointwise LTR approach both as a classification and regression task. However, for the final notebook and the 
report, we focused on the regression task. Both approaches can be performed by calling: 

```
python pointwise_class.py
python pointwise_reg.py
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
PointwiseLTR.ipynb
```

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

## Listwise Learning to Rank: LambdaRank 

LambdaRank is very similar to the spedup version of RankNet, which is why we used the same basic structure of the 
pairwiseSU.py file. However, as it is listwise, list based measures need to be performed within the training loss 
calculation. At first, we solved this with for-loops, which can be found in:

```
python listwise_noop_for_loops.py
```

As all of our files follow the same basic structure, running the scripts will yield the same as for the point- and pairwise
approaches. Due to time constraints, the hyperparameter search is trimmed and analogously to the pairwise approach, 
additionally to the number of hidden layers and learning rate, the sigma parameter is a hyperparameter. Furthermore, 
the IRM measure being NDCG or ERR is an additional hyperparameter.\

Though this script works, the slow training led us to come up with a vectorized version of the LambdaRank with NDCG, which 
can be computed with:

```
python listwise.py
```
 



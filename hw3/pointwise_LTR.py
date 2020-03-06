import dataset
import ranking as rnk
import evaluate as evl
import numpy as np
import torch


data = dataset.get_dataset().get_data_folds()[0]
data.read_data()



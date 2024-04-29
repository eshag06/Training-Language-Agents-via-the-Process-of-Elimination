import copy
import random
import time
import numpy as np
import torch

from util import DatasetLoader
from util import Query
from util import Model
from util import ExperimentLogger
from util import POE_threshold

model = Model(model_name='meta-llama/Llama-2-7b-hf')
experment = POE_threshold()

for dataset in ["MMLU"]:
    experment.experiment_with_modified_query(model, num_shots=5, dataset=dataset, decode_method='greedy', num_iter=25,
                                             seed=123,
                                             k=180)

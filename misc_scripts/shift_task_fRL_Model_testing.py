
from expanalysis.experiments.psychological_models import fRL_Model
import numpy as np
import pandas as pd
from selfregulation.utils.utils import get_behav_data

data = get_behav_data(file='Individual_Measures/shift_task.csv.gz')
workers = data.worker_id.unique()

# test divergence between hierarchical and flat experts after training
models = []
for worker in workers[0:5]:
    df = data.query("worker_id == '%s'" % worker)
    model = fRL_Model(df, decay_weights=True, verbose=True)
    model.optimize()
    models.append(model)




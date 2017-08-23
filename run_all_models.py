import numpy as np
from parameters import *
import model

task_list = ['dualDMS']
models_per_task = 25

for task in task_list:
    for j in range(0,models_per_task):
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)
        model.train_and_analyze()

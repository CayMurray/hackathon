## IMPORT MODULES ##

import pandas as pd
import numpy as np


## LOAD DATA ##

def load_labels(text_path):
    labels_list = []
    with open(text_path,'r') as f:
        for line in f:
            labels_list.append(line.split('xxx')[0])

    return pd.DataFrame(labels_list,columns=['labels'])


def load_data(array_path):
    array = np.load(array_path)

    return array.transpose(2,0,1)

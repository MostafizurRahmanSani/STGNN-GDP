import torch
import random
import numpy as np
import os

PAST_WINDOW = 5
HORIZON = 3

FIRST_YEAR = 1996
LAST_YEAR = 2019

TRAIN_END = 2009
VAL_END = 2012

DOWNLOAD_PREFIX = "https://raw.githubusercontent.com/pboennig/gnns_for_gdp/master"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDNN_DETERMINISTIC'] = '1'  
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)  
    os.environ['PYTHONHASHSEED'] = str(seed)
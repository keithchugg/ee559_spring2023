# utils.py

import numpy as np

def classification_error(gt, predicted):
    return np.sum(predicted != gt) / gt.shape[0] * 100

def classification_accuracy(gt, predicted):
    return np.sum(predicted == gt) / gt.shape[0] * 100




    
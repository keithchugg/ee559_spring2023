# ml_algorithms.py

import numpy as np 

class NearestMeansClassifier:
    def __init__(self):
        self.means = None

    def fit(self, data, label):
        self.means = np.zeros((np.unique(label).shape[0],data.shape[1]))
        for class_id in range(np.unique(label).shape[0]):
            self.means[class_id] = np.mean(data[label==class_id],axis=0)
        
    def predict(self, data):
        distances = np.zeros((data.shape[0],self.means.shape[0]))
        for class_id in range(distances.shape[1]):
            distances[:,class_id] = np.sqrt(np.sum((data - self.means[class_id,:])**2,axis=1))
        return np.argmin(distances,axis=1)






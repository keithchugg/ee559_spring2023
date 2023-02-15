import numpy as np
from ml_algorithms import NearestMeansClassifier
from utils import classification_error, classification_accuracy

def main():

    # Load data
    data = np.loadtxt("train.csv",delimiter=",", dtype=np.float64)
    train_data = data[:,:2]
    train_label = data[:,2]

    data = np.loadtxt("test.csv",delimiter=",", dtype=np.float64)
    test_data = data[:,:2]
    test_label = data[:,2]

    # Create object of the algorithm class and train, predict
    nmc = NearestMeansClassifier()
    nmc.fit(train_data,train_label)
    pred_train = nmc.predict(train_data)
    pred_test = nmc.predict(test_data)

    # Calculate error
    train_error =  classification_error(train_label,pred_train)
    test_error =  classification_error(test_label,pred_test)

if __name__=="__main__":
    main()














    
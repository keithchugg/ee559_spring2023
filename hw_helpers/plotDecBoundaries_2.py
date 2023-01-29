################################################
## EE559 HW1, Prof. Jenkins
## Created by Arindam Jati
## Updated by Thanos Rompokos
## Tested in Python 3.9.15
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries_2(training, label_train, sample_mean, fsize=(6,4),legend_on = False):
    
    '''
    Plot the decision boundaries and data points for minimum distance to
    class mean classifier
    
    training: traning data, N x d matrix:
        N: number of data points
        d: number of features 
        if d > 2 then the first and second features will be plotted (1st and 2nd column (0 and 1 index));
                 recommended to input an Nx2 dataset with the 2 columns of the features to be plotted
    label_train: class lables correspond to training data, N x 1 array:
        N: number of data points
    sample_mean: mean vector for each class, C x d matrix:
        C: number of classes (up to 10 classes the way the plot symbos are defined)
        each row of the sample_mean matrix is the coordinate of each sample mean
    legend_on: add the legend in the plot. Potentially slower for datasets with large number of clases and data points
        or would occupy significant part of the plot if too many classes
    '''

    #
    # Total number of classes
    nclass =  len(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.05

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    dist_mat = cdist(xy, sample_mean)
    pred_label = np.argmin(dist_mat, axis=1)

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    # documemtation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    symbols_ar = np.array(['rx', 'bo', 'ms', 'cd','gp','y*','kx','gP','r+','bh'])
    mean_symbol_ar = np.array(['rd', 'bd', 'md', 'cd','gd','yd','kd','gd','rd','bd'])
    markerfacecolor_ar = np.array(['r', 'b', 'm', 'c','g','y','k','g','r','b'])
    #show the image, give each coordinate a color according to its class label
    plt.figure(figsize=fsize)

    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower', aspect='auto')

    # plot the class training data.
    plot_index = 0
    class_list = []
    class_list_name = [] #for legend
    mean_list = [] # for legend
    mean_lis_name = [] # for legend
    for cur_label in np.unique(label_train):
        # print(cur_label,plot_index,np.sum(label_train == cur_label))
        d1, = plt.plot(training[label_train == cur_label, 0],training[label_train == cur_label, 1], symbols_ar[plot_index])

        if legend_on:
            class_list.append(d1)
            class_list_name.append('Class '+str(plot_index))
            l = plt.legend(class_list,class_list_name, loc=2)
            plt.gca().add_artist(l)

        # plot the class mean vector.
        m1, = plt.plot(sample_mean[cur_label,0], sample_mean[cur_label,1], mean_symbol_ar[plot_index], markersize=12, markerfacecolor=markerfacecolor_ar[plot_index], markeredgecolor='w')
        # include legend for class mean vector
        if legend_on:
            mean_list.append(m1)
            mean_lis_name.append('Class '+str(plot_index)+' mean')
            l1 = plt.legend(mean_list,mean_lis_name, loc=4)
            plt.gca().add_artist(l1)
       
        plot_index = plot_index + 1

    plt.show()
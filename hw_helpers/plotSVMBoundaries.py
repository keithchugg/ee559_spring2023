################################################
## EE559
## Created by Arindam Jati
## Tested in Python 3.6.3, OSX El Captain
################################################

import numpy as np
import matplotlib.pyplot as plt

def plotSVMBoundaries(training, label_train, classifier, support_vectors = [], fsize=(6,4),legend_on = True):
    #Plot the decision boundaries and data points for minimum distance to
    #class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # classifier: sklearn classifier model, must have a predict() function
    #
    # Total number of classes
    nclass =  max(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 0.01
    min_x = np.floor(min(training[:, 0])) - 0.01
    max_y = np.ceil(max(training[:, 1])) + 0.01
    min_y = np.floor(min(training[:, 1])) - 0.01

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
    pred_label = classifier.predict(xy)
    
    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    # documemtation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    symbols_ar = np.array(['rx', 'bo', 'ms', 'cd','gp','y*','kx','gP','r+','bh'])
    mean_symbol_ar = np.array(['rd', 'bd', 'md', 'cd','gd','yd','kd','gd','rd','bd'])
    markerfacecolor_ar = np.array(['r', 'b', 'm', 'c','g','y','k','g','r','b'])

    #turn on interactive mode
    plt.figure(figsize=fsize)
    # plt.ion()

    #show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    if len(support_vectors)>0:
        sv_x = support_vectors[:, 0]
        sv_y = support_vectors[:, 1]
        plt.scatter(sv_x, sv_y, s = 100, c = 'green')
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
       
        plot_index = plot_index + 1
    
        # # plot support vectors


    plt.show()

    # unique_labels = np.unique(label_train)
    # # plot the class training data.
    # plt.plot(training[label_train == unique_labels[0], 0],training[label_train == unique_labels[0], 1], 'rx')
    # plt.plot(training[label_train == unique_labels[1], 0],training[label_train == unique_labels[1], 1], 'go')
    # if nclass == 3:
    #     plt.plot(training[label_train == unique_labels[2], 0],training[label_train == unique_labels[2], 1], 'b*')

    # # include legend for training data
    # if nclass == 3:
    #     l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    # else:
    #     l = plt.legend(('Class 1', 'Class 2'), loc=2)
    # plt.gca().add_artist(l)

    # # plot support vectors
    # if len(support_vectors)>0:
    #     sv_x = support_vectors[:, 0]
    #     sv_y = support_vectors[:, 1]
    #     plt.scatter(sv_x, sv_y, s = 100, c = 'blue')

    # plt.show()
    


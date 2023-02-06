import numpy as np 
import matplotlib.pyplot as plt

def plot_multiclass_histograms(X_aug, W, y, fname, norm_W=False, scale=1, class_names=None):
    """
    Keith Chugg, USC, 2023.

    X_aug: shape: (N, D + 1).  Augmented data matrix
    W: shape: (D + 1, C).  The matrix of augmented weight-vectors.  W.T[m] is the weight vector for class m
    y: length N array with int values with correct classes.  Classes are indexed from 0 up.
    fname: a pdf of the histgrams will be saved to filename fname
    norm_W: boolean.  If True, the w-vectors for each class are normalized.
    scale: use scale < 1 to make the figure smaller, >1 to make it bigger
    class_names: pass a list of text, descriptive names for the classes.  

    This function takes in the weight vectors for a linear classifier and applied the "maximum value methd" -- i.e., 
    it computes the argmax_m g_m(x), where g_m(x) = w_m^T x to find the decision. For each class, it plots the historgrams 
    of  g_m(x) when class c is true.  This gives insights into which classes are most easily confused -- i.e., similar to a 
    confusion matrix, but more information.  

    Returns: the overall misclassification error percentage
    """
    if norm_W:
       W = W / np.linalg.norm(W, axis=0)
    y_soft = X_aug @ W
    N, C = y_soft.shape
    y_hard = np.argmax(y_soft, axis=1)
    error_percent = 100 * np.sum(y != y_hard) / len(y) 

    fig, ax = plt.subplots(C, sharex=True, figsize=(12 * scale, 4 * C * scale))
    y_soft_cs = []
    conditional_error_rate = np.zeros(C)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(C)]
    for c_true in range(C):
        y_soft_cs.append(X_aug[y_train == c_true] @ W)
        y_hard_c = np.argmax(y_soft_cs[c_true], axis=1)
        conditional_error_rate[c_true] = 100 * np.sum(y_hard_c != c_true) / len(y_hard_c)
    for c_true in range(C):
        peak = -100
        for c in range(C):
            hc = ax[c_true].hist(y_soft_cs[c_true].T[c], bins = 100, alpha=0.4, label=class_names[c])
            peak = np.maximum(np.max(hc[0]), peak)
            ax[c_true].legend()
            ax[c_true].grid(':')
        ax[c_true].text(0, 0.9 * peak, f'True: {class_names[c_true]}\nConditional Error Rate = {conditional_error_rate[c_true] : 0.2f}%')
    if norm_W:
        ax[C-1].set_xlabel(r'nromalized discriminant function $g_m(x) / || {\bf w} ||$')
    else:
        ax[C-1].set_xlabel(r'discriminant function $g_m(x)$')
    plt.savefig(fname, bbox_inches='tight',)
    return error_percent
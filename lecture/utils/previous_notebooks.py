import numpy as np 
import matplotlib.pyplot as plt


def generate_colored_nongaussian_data(means, lambdas, thetas, Ns, distribution='normal', quiet_mode='true'):
    """
    means: shape (2, 2), means[0] is the 2 x 1 mean vector for class 1 data generation
    lambdas: shape (2, 2), lambdas[0] are the 2 eigenvalues of the covariance matrix for generatinge data for class 1
    Ns: [N1, N2] the number of samples to be generated for each of teh two classes.
    distribution: in {normal, exponential, uniform} sets the distribution to generate data for both classes.
    quiet_mode: added this so that it won't print the details unless quiet_mode == False
    """
    N1 = Ns[0]
    N2 =  Ns[1]
    N = N1 + N2
    x = np.zeros((N, 2))
    assert distribution in {'normal', 'exponential', 'uniform'}, f'The {distribution} is not supported, only normal, exponential, uniform distributions are supported.'
    assert np.min(lambdas) > 0, f'lambda all have to be > 0 as they are variaces of the random vector projected onto the eigen-vectors.  You passed lambdas = {lambdas}'
    if distribution == 'normal':
        x[:N1] = np.random.normal(0, 1, (N1, 2))
        x[N1:] = np.random.normal(0, 1, (N2, 2))
    elif distribution == 'exponential':
        ## np.random.exponential(1) generates realizations from a unit variance, mean 1
        x[:N1] = np.random.exponential(1, (N1, 2)) - 1
        x[N1:] = np.random.exponential(1, (N2, 2)) - 1
    elif distribution == 'uniform':
        ## variance of uniform on (a,b) is (b-a)^2 / 12
        a = np.sqrt(3)
        x[:N1] = np.random.uniform(-a, a, (N1, 2))
        x[N1:] = np.random.uniform(-a, a, (N1, 2))

    def compute_coloring_matrix(theta, lams):
        E = np.asarray([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ])
        Lambda_root = np.sqrt( np.asarray([ [lams[0], 0], [0, lams[1]] ]) )
        H = E @ Lambda_root
        K = H @ H.T
        return H, K

    H1, K1 = compute_coloring_matrix(thetas[0], lambdas[0])
    H2, K2 = compute_coloring_matrix(thetas[1], lambdas[1])

    x[:N1] = x[:N1] @ H1.T + means[0]
    x[N1:] = x[N1:] @ H2.T + means[1]

    labels = np.ones(N)
    labels[N1:] += 1

    sample_means = np.zeros((2,2))
    sample_means[0] = np.mean(x[:N1], axis=0)
    sample_means[1] = np.mean(x[N1:], axis=0)

    if not quiet_mode:
        print(f'Data generated under the {distribution} distribution')
        Ks = [K1, K2]
        Hs = [H1, H2]

        for i in range(2):
            print(f'The mean in the generating pdf for class {i + 1} is: {means[i]}')
            print(f'The sample mean for class {i + 1}  data is: {sample_means[i]}\n')

            print(f'The coloring matrix class {i + 1}  data is:\n {Hs[i]}')
            print(f'The covariance matrix class {i + 1}  data is:\n {Ks[i]}\n\n')

    return x, labels, sample_means

def solve_plot_ls_nm_classifier(x, labels):
    ## LS classifier
    N = x.shape[0]
    X_tilde = np.ones((N, 3))   ## the feature vector is dimension 2, and this is the extended version   
    X_tilde[:, 1:] = x          ## the first column is all 1s, this sets the rest of each row to the data samples
    y = (-1.0) ** (labels + 1)        ## (-1)^(1 + 1) = +1 and (-1)^(2+1) = -1, maps class 1 and class 2 to +1, -1, resp.
    w_ls, Re, rank, singular_vals = np.linalg.lstsq(X_tilde, y, rcond=None)

    ## Nearest Means Classifier:
    x_1 = x[labels==1]
    x_2 = x[labels==2]
    mu1 = np.mean(x_1, axis=0)
    mu2 = np.mean(x_2, axis=0)
    w_nm = np.ones(3)
    w_nm[0] = 0.5 * (np.dot(mu2, mu2) - np.dot(mu1, mu1))
    w_nm[1:] = mu1 - mu2

    plt.figure(figsize=(6, 6))
    LIMIT = np.max(x)
    x_plot = np.arange(-1 * LIMIT, LIMIT, 0.01)
    plt.scatter(x_1.T[0], x_1.T[1], fc=(0, 0, 1, 0.5), label='class 1')
    plt.scatter(x_2.T[0], x_2.T[1], fc=(1, 0, 0, 0.5), label='class 2')
    ## plot the decision boundaries which is g(x) = 0
    plt.plot( x_plot, -1 * ( w_ls[1] *  x_plot  + w_ls[0] ) / w_ls[2], linewidth=3, c='k', label='LS boundary')
    plt.plot( x_plot, -1 * ( w_nm[1] *  x_plot  + w_nm[0] ) / w_nm[2], linewidth=3, c='g', label='NM boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim([-LIMIT, LIMIT])
    plt.ylim([-LIMIT, LIMIT])
    plt.legend()
    plt.grid(':')

    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(18, 4))
    titles = ['Least Squares Classifier Discriminant Histogram', 'Nearest Meanss Classifier Discriminant Histogram']
    for i, w in enumerate([w_ls, w_nm]):
        g1 = x_1 @ w[1:] + w[0] ## discriminate function values for all data in class 1
        g2 = x_2 @ w[1:] + w[0] ## discriminate function values for all data in class 2
        h1 = ax[i].hist(g1, bins = 100, fc=(0, 0, 1, 0.5), label='class 1')
        h2 = ax[i].hist(g2, bins = 100, fc=(1, 0, 0, 0.5), label='class 2')
        N1_errors = np.sum(g1 < 0)  ## error condition:  g > 0 <==> x in Gamma_1
        N2_errors = np.sum(g2 >= 0) ## error condition:  g <= 0 <==> x in Gamma_2
        error_rate = (N1_errors + N2_errors) / N
        ax[i].axvline(0, linewidth=0.5, linestyle='dashed', color='k')
        
        ax[i].grid(':')
        ax[i].legend()
        ax[i].set_xlabel('discriminant function g(x)')
        ax[i].set_ylabel('histogram count')
        ax[i].set_title(titles[i])
        peak = np.maximum(np.max(h1[0]), np.max(h2[0]))
        ax[i].text(0, 0.7 * peak, f'Error rate = {error_rate : 0.3f}% ({N1_errors + N2_errors}/{N})')
    

    return w_ls, w_nm
    print(f'\nw vector for LS: {w_ls}')
    print(f'w vector for NM: {w_nm}\n')
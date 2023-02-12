import numpy as np 
import matplotlib.pyplot as plt

N = 8
eta = 0.5
N_epochs = 2

w = np.asarray([-2, 4])
x = np.random.choice([-1 , +1], 2 * N).reshape(N, 2)
w_hat = np.zeros(2)

w_hat_hist = np.zeros((N * N_epochs + 1, 2))
w_hat_hist[0] = w_hat[:]
print(f'\nInitial w_hat: {w_hat}\n')
for i in range(N * N_epochs):
    y = np.dot(w, x[i % N])
    y_hat = np.dot(w_hat, x[i % N])
    error = y_hat - y
    w_hat = w_hat - eta * error * x[i % N]
    w_hat_hist[i+1] = w_hat
    print(f'i = {i}\tx_n = {x[i % N]}')
    print(f'i = {i}\ty_n = {y : 0.2f}\ty_hat = {y_hat : 0.2f}\terror = {error : 0.2f}')
    print(f'i = {i}\tw_hat[{i}] = {w_hat}\n')

plt.figure()
iterations = np.arange(N * N_epochs + 1)
plt.plot(iterations, w_hat_hist.T[0], color='g', linestyle = '--', marker='o', label=r'$\hat{w}_0$')
plt.plot(iterations, w_hat_hist.T[1], color='b', linestyle = '--', marker='o', label=r'$\hat{w}_1$')
plt.axhline(w[0], c='g', )
plt.axhline(w[1], c='b')
plt.xlabel('iteration')
plt.ylabel('w coefficents and estimates')
plt.grid(':')
plt.legend()
plt.show()
plt.close()
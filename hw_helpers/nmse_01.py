def normalized_mse_01(f, f_hat, x_grid, G=10000):
    x_fine =  np.linspace(0, 1, G)
    f_fine = f(x_fine)
    f_hat_fine = np.interp(x_fine, x_grid, f_hat)
    sq_error = (f_fine - f_hat_fine) ** 2
    mse = np.mean(sq_error)
    ref = np.mean(f_fine ** 2)
    return mse / ref
r2_score_1:
layers = [X_train.shape[1], 50, Y_train.shape[1]]
activation_func = ['sigmoid', 'nothing']
epochs = 50
batch_size = 200
eta_vals = np.logspace(-4, -8, 6)
lmbda_vals = np.logspace(0, -5, 5)
lmbda_vals[-1] = 0

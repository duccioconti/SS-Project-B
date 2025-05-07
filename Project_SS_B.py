# Project SS B

import numpy as np

# initialization
C = np.array ([[4, -1], [-1, 4]])
C_inv = np.array ([[4/15, 1/15], [1/15, 4/15]])
mean = [0, 0]

a = [1, 3, 10]
n = 100000

Y = np.zeros (n)
L = np.zeros (3)
U = np.zeros (3)

# generating standard normal r.v.
seed = 42
gen = np.random.default_rng(seed)
Z = gen.multivariate_normal (mean, C, size=n)

# (a)
for i in range (3):
    # generate Y = 1 {Z that belongs to A}
    Y = np.logical_and (Z[:, 0]>a[i], Z[:, 1]>a[i])

    # calculate the mean estimator of Y
    mu = np.mean(Y)

    # calculate the standard deviation estimator of Y
    X = (Y - mu)**2
    sd = np.sqrt ( sum (X) / (n-1) )

    # calculate the Lower bound of the 95% confidence interval
    L[i] = mu - 1.96 * sd / np.sqrt(n)

    # calculate the Lower bound of the 95% confidence interval
    U[i] = mu + 1.96 * sd / np.sqrt(n)

# print the 95% confidence interval for each a
for i in range(3):
    print (f'The 95% confidence interval with a = {a[i]} is [{L[i]}, {U[i]}]')

# (b)
for i in range (3):
    b = np.array ([a[i], a[i]])
    # generating r.v. distributed as g
    seed = 42
    gen = np.random.default_rng(seed)
    Z_shift = gen.multivariate_normal (b, C, size=n)

    Lx = np.exp (- b @ C_inv @ Z_shift.T + 0.5 * (b @ C_inv @ b.T).item() )
    
    # generate Y = 1 {Z that belongs to A}
    Y = np.logical_and (Z_shift[:, 0]>a[i], Z_shift[:, 1]>a[i])

    # calculate the mean estimator of Y
    mu = np.mean(Y*Lx)

    # calculate the standard deviation estimator of Y
    X = (Y*Lx - mu)**2
    sd = np.sqrt ( sum (X) / (n-1) )

    # calculate the Lower bound of the 95% confidence interval
    L[i] = mu - 1.96 * sd / np.sqrt(n)

    # calculate the Lower bound of the 95% confidence interval
    U[i] = mu + 1.96 * sd / np.sqrt(n)

# print the 95% confidence interval for each a
for i in range(3):
    print (f'The 95% confidence interval with a = {a[i]} is [{L[i]}, {U[i]}]')

# (c)
delta = 1
C = delta*C
C_inv = np.linalg.inv (C)

for i in range (3):
    b = np.array ([a[i], a[i]])
    # generating r.v. distributed as g
    seed = 42
    gen = np.random.default_rng(seed)
    Z_shift = gen.multivariate_normal (b, C, size=n)

    Lx = np.exp (- b @ C_inv @ Z_shift.T + 0.5 * (b @ C_inv @ b.T).item() )
    
    # generate Y = 1 {Z that belongs to A}
    Y = np.logical_and (Z_shift[:, 0]>a[i], Z_shift[:, 1]>a[i])

    # calculate the mean estimator of Y
    mu = np.mean(Y*Lx)

    # calculate the standard deviation estimator of Y
    X = (Y*Lx - mu)**2
    sd = np.sqrt ( sum (X) / (n-1) )

    # calculate the Lower bound of the 95% confidence interval
    L[i] = mu - 1.96 * sd / np.sqrt(n)

    # calculate the Lower bound of the 95% confidence interval
    U[i] = mu + 1.96 * sd / np.sqrt(n)

# print the 95% confidence interval for each a
for i in range(3):
    print (f'The 95% confidence interval with a = {a[i]} is [{L[i]}, {U[i]}]')
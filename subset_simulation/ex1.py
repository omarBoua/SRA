import numpy as np
def g(c1, c2, m, r, t1, F1):
    w0 = np.sqrt((c1 * c2)/m)
    k = 3
    return k * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))
def gcal1(X,mu,sdev):
    X = mu + sdev * X 
    omega_2 = (X(2) + X(3))/X(1)
    return 3*X(4) - abs(2*X(5)/(X(1)*omega_2)*np.sin(0.5*np.sqrt(omega_2)*X(6)))


dim = 6
mu = np.array([1, 1, 0.1, 0.5, 1, 1])  # mean of rvs
sdev = np.array([0.05, 0.1, 0.01, 0.05, 0.2, 0.2])  # sdev of rvs

Nsim = 1
Nlevel = 500
p0 = 0.1

for k in range(Nsim):
    U = np.random.normal(0,1,size = (Nlevel,dim))
    g = np.zeros(Nlevel)
    for i in range(Nlevel):
        g[i] = gcal1(U[i],mu,sdev)


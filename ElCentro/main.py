import numpy as np
import scipy.io
import time as tm

# Load the data from the .mat files
elcentro_ns = scipy.io.loadmat('ElcentroNS.mat')
time_NS = elcentro_ns['e000'].flatten()
NSacc = elcentro_ns['e003'].flatten()

elcentro_ew = scipy.io.loadmat('ElcentroEW.mat')
time_EW = elcentro_ew['e000'].flatten()
EWacc = elcentro_ew['e004'].flatten()

# Adjust the length of EWacc to match time_NS
len_diff = len(time_NS) - len(time_EW)
EWacc = np.concatenate((EWacc, np.zeros(len_diff)))
time_array = time_NS
delta = 0.02
n_T = len(time_array)

# System parameters
ma = 10000 * np.ones(10)
eta = 0.04 * np.ones(10)
st = np.array([40, 40, 40, 36, 36, 36, 32, 32, 32, 32]) * 1000000
da = (2 * eta) * np.sqrt(ma * st)
C = np.zeros((10, 10))
M = np.zeros((10, 10))
K = np.zeros((10, 10))
for s in range(10):
    M[s, s] = ma[s]
    if s == 0:
        C[s, s] = da[s] + da[s + 1]
        K[s, s] = st[s] + st[s + 1]
        C[s, s + 1] = -da[s + 1]
        K[s, s + 1] = -st[s + 1]
    elif s == 9:
        C[s, s] = da[s]
        K[s, s] = st[s]
        C[s, s - 1] = -da[s]
        K[s, s - 1] = -st[s]
    else:
        C[s, s] = da[s] + da[s + 1]
        K[s, s] = st[s] + st[s + 1]
        C[s, s + 1] = -da[s + 1]
        K[s, s + 1] = -st[s + 1]
        C[s, s - 1] = -da[s]
        K[s, s - 1] = -st[s]
Nsim = 100000

gamma = 0.5
beta = 0.25

A1 = -np.linalg.inv(M) @ C
A2 = -np.linalg.inv(M) @ K
TT = np.ones(10)
A3 = np.linalg.inv(M + gamma * delta * C + beta * delta ** 2 * K)

Cres = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, -1, 1]])
uu = np.random.randn(2, Nsim)

start_time = tm.time()

R1 = np.zeros(Nsim)
R2 = np.zeros(Nsim)
gg = np.zeros(Nsim)

for i in range(Nsim):
    gacc = (5 + 0.5 * uu[0, i]) * EWacc + (5 + 0.5 * uu[1, i]) * NSacc
    U_c = np.zeros(10)  # displacement x(t)
    Udot_c = np.zeros(10)  # velocity x'(t)
    R = np.zeros((2, n_T))
    for j in range(n_T - 1):
        R[:, j] = Cres @ U_c
        Uddot_c = A1 @ Udot_c + A2 @ U_c + TT * gacc[j]
        P = Udot_c + (1 - gamma) * delta * Uddot_c
        Q = U_c + delta * Udot_c + 0.5 * delta ** 2 * (1 - 2 * beta) * Uddot_c
        Uddot_n = A3 @ (TT * gacc[j + 1] - C @ P - K @ Q)
        U_n = U_c + delta * Udot_c + 0.5 * delta ** 2 * ((1 - 2 * beta) * Uddot_c + 2 * beta * Uddot_n)
        Udot_n = Udot_c + (1 - gamma) * delta * Uddot_c + gamma * delta * Uddot_n
        U_c = U_n
        Udot_c = Udot_n
        Uddot_c = Uddot_n
    R[:, -1] = Cres @ U_c
    R1[i] = 1000 * np.max(np.abs(R[0, :]))
    R2[i] = 1000 * np.max(np.abs(R[1, :]))
    gg[i] = np.min([10 - R1[i], 2.2 - R2[i]])

elapsed_time = tm.time() - start_time
pf = np.sum(gg > 0) / Nsim
print(pf)
print(f"Simulation time: {elapsed_time} seconds")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings

warnings.filterwarnings("ignore")


# Function to calculate the performance function using the given system parameters
def performance_function(uu, EWacc, NSacc, delta, n_T, Cres, A1, A2, TT, A3):
    gacc = (5 + 0.5 * uu[0]) * EWacc + (5 + 0.5 * uu[1]) * NSacc
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
    R1 = 1000 * np.max(np.abs(R[0, :]))
    R2 = 1000 * np.max(np.abs(R[1, :]))

    gg = np.min([10 - R1, 2.2 - R2])
    return gg


# Stage 1: Generation of Monte Carlo population
nMC = 1000000
uu1 = np.random.randn(2, nMC)
uu2 = np.random.randn(2, nMC)
S = np.column_stack((uu1, uu2))
function_calls = 0

# Stage 2: Definition of initial design of experiments (DoE)
N1 = 12
n_EDini = N1
selected_indices = np.random.choice(len(S), N1, replace=False)
DoE = S[selected_indices]

# Evaluate the performance function on the initial design of experiments (DoE)
Pf_values = np.zeros(N1)
for i in range(N1):
    Pf_values[i] = performance_function(DoE[i, :], EWacc, NSacc, delta, n_T, Cres, A1, A2, TT, A3)
    function_calls += 1

# Stage 3: Computation of Kriging model
scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)
kriging = GaussianProcessRegressor()
kriging.fit(scaled_DoE, Pf_values)
iter = 0
function_calls_values = []
pf_hat_values = []

while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    G_hat, kriging_std = kriging.predict(scaler.transform(S), return_std=True)
    Pf_hat = np.sum(G_hat < 0) / nMC

    # Stage 5: Identification of the best next point to evaluate
    learning_values = np.abs(G_hat) / kriging_std
    x_best_index = np.argmin(learning_values)
    x_best = S[x_best_index]

    # Stage 6: Stopping condition on learning
    stopping_condition = min(learning_values) >= 2

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat * nMC))
        cov_threshold = 0.05

        if cov_pf <= cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.4e}".format(Pf_hat))
            print("Coefficient of variation: {:.4%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
        else:
            # Coefficient of variation is too high, update population
            new_uu1 = np.random.randn(2, nMC)
            new_uu2 = np.random.randn(2, nMC)
            new_points = np.column_stack((new_uu1, new_uu2))
            S = np.vstack((S, new_points))
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = performance_function(x_best, EWacc, NSacc, delta, n_T, Cres, A1, A2, TT, A3)
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        scaled_DoE = scaler.fit_transform(DoE)
        kriging.fit(scaled_DoE, Pf_values)

    iter += 1
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)
    print("iter ", iter, ": ", Pf_hat)

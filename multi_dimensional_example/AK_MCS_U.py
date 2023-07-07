import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")

def g(X):
    n = len(X)
    sigma = np.std(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)

function_calls = 0
nMC = 300000 # Number of instances to generate
n = 100  # Number of parameters

mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n))







# Stage 2: Definition of initial design of experiments (DoE)
N1 = 50
n_EDini = N1 

mean_population = np.mean(S, axis=0)
distances_to_mean = cdist([mean_population], S)
closest_sample_index = np.argmin(distances_to_mean)
initial_design = [S[closest_sample_index]]

for _ in range(n_EDini - 1):
    distances_to_design = cdist(initial_design, S)
    farthest_sample_index = np.argmax(np.min(distances_to_design, axis=0))
    initial_design.append(S[farthest_sample_index])

DoE = np.array(initial_design)


""" selected_indices = np.random.choice(len(S), N1, replace=False)
DoE = S[selected_indices]
 """


Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = g(DoE[i])  # Evaluate performance function
    function_calls += 1


# Stage 3: Computation of Kriging model
scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)
#kernel = ConstantKernel(1.0) * RBF(1.0)
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e2)) # Decreased lower bound from 1e-2 to 1e-3
kriging = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

kriging.fit(scaled_DoE, Pf_values)
iter =0
function_calls_values = []
pf_hat_values = []
U_values_iter = []
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(scaler.transform(S),return_std=True)
    Pf_hat = np.sum(G_hat <= 0) / nMC
    
    # Stage 5: Identification of the best next point to evaluate
    learning_values = np.abs(G_hat) / kriging_std
    x_best_index = np.argmin(learning_values)
    x_best = S[x_best_index]
    # Stage 6: Stopping condition on learning
    U_values_iter.append(min(learning_values))
    stopping_condition = min(learning_values) >= 0.2
    print("std ", kriging_std[x_best_index]
           ) 
    print("G_hat", G_hat[x_best_index])
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)
    print(min(learning_values))

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        break
        # Stopping condition met, learning is stopped
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat* nMC) )
        print("cov", cov_pf)
        cov_threshold = 0.05
        if cov_pf < cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.2e}".format(Pf_hat))
            print("Coefficient of variation: {:.2%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
            # Stage 10: End of AK-MCS
        else:
            # Coefficient of variation is too high, update population
          
            new_points = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n)) 
            S = np.vstack((S, new_points))
            # Go back to Stage 4
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = g(x_best)
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        scaled_DoE = scaler.transform(DoE)
        kriging.fit(scaled_DoE, Pf_values)        
        # Go back to Stage 4
    iter += 1
    

    print("iter ",iter, ": ",Pf_hat)


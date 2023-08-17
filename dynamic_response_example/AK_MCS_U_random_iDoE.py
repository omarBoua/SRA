import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
warnings.filterwarnings("ignore")

#Performance function: set k = 1.5 for lower probability 
def g(c1, c2, m, r, t1, F1):
    w0 = np.sqrt((c1 * c2)/m)
    k = 3
    return k * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))

# Stage 1: Generation of Monte Carlo population
nMC = 70000    #set nMC = 100000 for lower probability
m = np.random.normal(1, 0.05, size=nMC)
c1 = np.random.normal(1, 0.1, size=nMC)
c2 = np.random.normal(0.1, 0.01, size=nMC)
r = np.random.normal(0.5, 0.05, size=nMC)
F1 = np.random.normal(1, 0.2, size=nMC)
t1 = np.random.normal(1, 0.2, size=nMC) 
S = np.column_stack((c1, c2, m, r, t1, F1))

# Stage 2: Definition of initial design of experiments (DoE)
N1 = 50
n_EDini = N1 
function_calls = 0
selected_indices = np.random.choice(len(S), N1, replace=False)
DoE = S[selected_indices] 
Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = g(DoE[i, 0], DoE[i, 1],DoE[i,2], DoE[i,3], DoE[i,4], DoE[i,5])  # Evaluate performance function
    function_calls += 1

# Stage 3: Computation of Kriging model
scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)
kernel = C(1.0, (1e-3, 1e3)) * RBF(np.repeat([0.5], 6), (1e-3, 1e2))  # Decreased lower bound from 1e-2 to 1e-3
kriging = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)
kriging.fit(scaled_DoE, Pf_values)
iter =0
function_calls_values = []
pf_hat_values = []
#start iterative process
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(scaler.transform(S),return_std=True)
    Pf_hat = np.sum(G_hat >= 0) / nMC
    
    # Stage 5: Identification of the best next point to evaluate
    learning_values = np.abs(G_hat) / kriging_std
    x_best_index = np.argmin(learning_values)
    x_best = S[x_best_index]

    # Stage 6: Stopping condition on learning
    stopping_condition = min(learning_values) >= 2  

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        # Stopping condition met, learning is stopped
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat* nMC) )
        cov_threshold = 0.05
        if cov_pf < cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.2e}".format(Pf_hat))
            print("Coefficient of variation: {:.2%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            # Stage 10: End of AK-MCS
            break
          
        else:
            # Coefficient of variation is too high, update population
            new_m = np.random.normal(1, 0.05, size=nMC)
            new_c1 = np.random.normal(1, 0.1, size=nMC)
            new_c2 = np.random.normal(0.1, 0.01, size=nMC)
            new_r = np.random.normal(0.5, 0.05, size=nMC)
            new_F1 = np.random.normal(1, 0.2, size=nMC)
            new_t1 = np.random.normal(1, 0.2, size=nMC) 
            new_points = np.column_stack((new_c1, new_c2, new_m, new_r, new_t1, new_F1)) 
            S = np.vstack((S, new_points))
            # Go back to Stage 4
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = g(x_best[0], x_best[1],x_best[2], x_best[3], x_best[4], x_best[5])
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        scaled_DoE = scaler.transform(DoE)
        kriging.fit(scaled_DoE, Pf_values)
        # Go back to Stage 4
    iter += 1
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)
    print("iter ",iter, ": pf value is ",Pf_hat)


# Plotting pf_hat values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'b-')
plt.xlabel('function_calls')
plt.ylabel('pf_hat')
plt.title('Convergence Plot')


# Display the number of iterations
plt.text(0.95, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show() 
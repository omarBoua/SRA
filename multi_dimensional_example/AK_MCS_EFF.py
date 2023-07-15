import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")

def eff(g_hat_values, sigma_g_values):
    a = 0
    epsilon = 2 * np.square(sigma_g_values)

    term1 = (g_hat_values - a) * (2 * norm.cdf((a - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a + epsilon - g_hat_values) / sigma_g_values))
    term2 = -sigma_g_values * (2 * norm.pdf((a - g_hat_values) / sigma_g_values) - norm.pdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.pdf((a + epsilon - g_hat_values) / sigma_g_values))
    term3 = norm.cdf((a + epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values)

    eff_values = term1 + term2 + term3
    return eff_values
def g(X):
    n = len(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)

function_calls = 0
nMC = 300000 # Number of instances to generate
n = 40  # Number of parameters

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
kernel = C(1, (1e-3, 1e2)) * RBF(10, (1e-2, 1e2))  # Decreased lower bound from 1e-2 to 1e-3
kriging = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30) #30 for n= 40 was good

kriging.fit(scaled_DoE, Pf_values)
iter =0
function_calls_values = []
pf_hat_values = []
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(scaler.transform(S),return_std=True)
    Pf_hat = np.sum(G_hat <= 0) / nMC
    
    # Stage 5: Identification of the best next point to evaluate
    learning_values = eff(G_hat, kriging_std)
    x_best_index = np.argmax(learning_values)
    x_best = S[x_best_index]
    # Stage 6: Stopping condition on learning
    stopping_condition = max(learning_values) <= 0.001
    print("std ", kriging_std[x_best_index]
           ) 
    print("G_hat", G_hat[x_best_index])
    print(max(learning_values))
    print("iter ",iter, ": ",Pf_hat)
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        # Stopping condition met, learning is stopped
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat* nMC) )
        print("cov", cov_pf)
        cov_threshold = 0.05
        if cov_pf <= cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.4e}".format(Pf_hat))
            print("Coefficient of variation: {:.4%}".format(cov_pf))
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
    

""" 

x1_vals = np.linspace(-6, 6, 1000)
x2_vals = np.linspace(-6, 6, 1000)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Calculate LSF values for each combination of x1 and x2
Z = np.array([performance_function(x1, x2) for x1, x2 in zip(X1.flatten(), X2.flatten())])
Z = Z.reshape(X1.shape)

# Plotting the contour of LSF
plt.contour(X1, X2, Z, levels=[0], colors='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('LSF Contour')

# Plotting the initial points in the design of experiment
plt.scatter(DoE[:, 0], DoE[:, 1], c='blue', s=5, label='Initial Points', marker = 'o')

# Plotting the added points in the final design of experiment
plt.scatter(DoE[n_EDini:, 0], DoE[n_EDini:, 1], c='red',s=5, label='Added Points', marker = 'o')


legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=1, label='G = 0'),
    plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=5, label='Initial Points'),
    plt.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=5, label='Added Points')
]


plt.legend(handles=legend_elements)

plt.show() """


# Plotting pf_hat values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'b-')
plt.xlabel('function_calls')
plt.ylabel('pf_hat')
plt.title('Convergence Plot')


# Indicate the last point
last_point_calls = function_calls_values[-1]
last_point_pf_hat = pf_hat_values[-1]
plt.plot(last_point_calls, last_point_pf_hat, 'ro')
plt.annotate(f'({last_point_calls}, {last_point_pf_hat})',
             xy=(last_point_calls, last_point_pf_hat),
             xytext=(last_point_calls  , last_point_pf_hat+last_point_pf_hat/10 ),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Display the number of iterations
plt.text(0.95, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show() 
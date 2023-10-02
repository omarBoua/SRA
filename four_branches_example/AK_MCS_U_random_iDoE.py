import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

def performance_function(x1, x2):
    k = 6
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)
# Stage 1: Generation of Monte Carlo population
nMC = 1000000
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))
function_calls = 0





# Stage 2: Definition of initial design of experiments (DoE)
N1 = 12
#uncomment for random selection
n_EDini = N1 
selected_indices = np.random.choice(len(S), N1, replace=False)

DoE = S[selected_indices]

""" #uncomment for importance sampling
mean_population = np.mean(S, axis=0)
distances_to_mean = cdist([mean_population], S)
closest_sample_index = np.argmin(distances_to_mean)
initial_design = [S[closest_sample_index]]

for _ in range(N1 - 1):
    distances_to_design = cdist(initial_design, S)
    farthest_sample_index = np.argmax(np.min(distances_to_design, axis=0))
    initial_design.append(S[farthest_sample_index])

DoE = np.array(initial_design)
"""
Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = performance_function(DoE[i, 0], DoE[i, 1])  # Evaluate performance function
    function_calls += 1


# Stage 3: Computation of Kriging model
scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)
#kernel = ConstantKernel(1.0) * RBF(1.0)
kriging = GaussianProcessRegressor()
kriging.fit(scaled_DoE, Pf_values)
iter =0
function_calls_values = []
pf_hat_values = []
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(scaler.transform(S),return_std=True)
    Pf_hat = np.sum(G_hat < 0) / nMC
    
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
        if cov_pf <= cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.4e}".format(Pf_hat))
            print("Coefficient of variation: {:.4%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
            # Stage 10: End of AK-MCS
        else:
            # Coefficient of variation is too high, update population
            new_x = np.random.normal(0,1, nMC)
            new_y = np.random.normal(0,1,nMC)
            new_points = np.column_stack((new_x, new_y)) 
            S = np.vstack((S, new_points))
            # Go back to Stage 4
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = performance_function(x_best[0], x_best[1])
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        scaled_DoE = scaler.transform(DoE)
        kriging.fit(scaled_DoE, Pf_values)
        # Go back to Stage 4
    iter += 1
    function_calls_values.append(function_calls)
    pf_hat_values.append(Pf_hat)
    print("iter ",iter, ": ",Pf_hat)



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

plt.show() 

""" 
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

plt.show()  """
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def performance_function(x1,x2):
    return x2 - np.abs(np.tan(x1)) - 1


# Stage 1: Generation of Monte Carlo population
nMC = 1000000
x1 = np.random.normal(4, 2, size=nMC)
x2 = np.random.normal(4, 2, size=nMC)
S = np.column_stack((x1, x2))
function_calls = 0

MCS_pf = 0.7007
# Stage 2: Definition of initial design of experiments (DoE)
N1 = 500
selected_indices = np.random.choice(len(S), N1, replace=False)

DoE = S[selected_indices]
Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = 1 if performance_function(DoE[i, 0], DoE[i, 1]) <0 else 0 # Evaluate performance function
    function_calls += 1


# Stage 3: Computation of MLP model


mlp = MLPClassifier(hidden_layer_sizes=10, activation= 'logistic', solver = 'adam',learning_rate = 'adaptive', learning_rate_init= 0.01)  # Customize the hidden layer sizes as needed
mlp.fit(DoE, Pf_values)

while True:
    # Stage 4: Prediction by MLP and estimation of probability of failure
    nMC = len(S)
    classes_hat = mlp.predict(S)
    
    y_pred_class = np.where(classes_hat > 0.5, 1, 0)
    Pf_hat = np.sum(y_pred_class == 0) / nMC
    # Stage 5: Identification of the best next point to evaluate based on distance to the threshold 0.5
    learning_values = np.abs(np.subtract(classes_hat, 0.5))
    x_best_index = np.argmin(learning_values)
    x_best = S[x_best_index]
    
    # Stage 6: Stopping condition on learning
    stopping_condition = min(learning_values) >= 0.1

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        # Stopping condition met, learning is stopped
        diff_prob_percentage = (np.abs(Pf_hat - MCS_pf) / MCS_pf) *100
        print(diff_prob_percentage)
        cov_threshold = 3
        if diff_prob_percentage < cov_threshold:
            

            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: ", Pf_hat)
            print("Number of calls to the performance function", function_calls)
            break
        else:
            # Coefficient of variation is too high, update population
            new_x = np.random.normal(4, 4, nMC)
            new_y = np.random.normal(4, 4, nMC)
            new_points = np.column_stack((new_x, new_y)) 
            S = np.vstack((S, new_points))
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = 1 if performance_function(x_best[0], x_best[1]) <0 else 0
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        
        mlp.fit(DoE, Pf_values)
        # Go back to Stage 4

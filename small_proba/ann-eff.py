import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import math
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

def eff(g_hat_values, sigma_g_values):
    a = 0
    epsilon = 2 * np.square(sigma_g_values)

    term1 = (g_hat_values - a) * (2 * norm.cdf((a - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a + epsilon - g_hat_values) / sigma_g_values))
    term2 = -sigma_g_values * (2 * norm.pdf((a - g_hat_values) / sigma_g_values) - norm.pdf((a - epsilon - g_hat_values) / sigma_g_values) - norm.pdf((a + epsilon - g_hat_values) / sigma_g_values))
    term3 = norm.cdf((a + epsilon - g_hat_values) / sigma_g_values) - norm.cdf((a - epsilon - g_hat_values) / sigma_g_values)

    eff_values = term1 + term2 + term3
    return eff_values
# Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def performance_function(x1, x2):
    global function_calls
    function_calls += 1
    return 0.5 * (x1 - 2) ** 2 - 1.5 * (x2 - 5) ** 3 - 3


function_calls = 0

nMC = 1000000
x1 = np.random.normal(0, 1, size=nMC)
x2 = np.random.normal(0, 1, size=nMC)
S = np.column_stack((x1, x2))

# 2. create the initial design of experimental
n_EDini = 12
selected_indices = np.random.choice(len(S), n_EDini, replace=False)

DoE = np.array(S[selected_indices])

initial_design = np.array(DoE)
labels = np.zeros(n_EDini)
for i in range(n_EDini):
    labels[i] = performance_function(initial_design[i, 0],
                                     initial_design[i, 1])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])  # smoothing the labels

scaler = StandardScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)

# Create a k-fold cross validator
n_splits = 5

scaled_S = scaler.transform(S)

models = []
for _ in range(n_splits):
    model = MLPRegressor(hidden_layer_sizes=(20), max_iter=100000, activation='tanh', solver='lbfgs',
                         early_stopping=True)
    models.append(model)

base_model = MLPRegressor(hidden_layer_sizes=(20), max_iter=100000, activation='tanh', solver='lbfgs',
                          early_stopping=True)
iter = 0
kf = KFold(n_splits=n_splits)

i = 0
while True:

    predictions = [[] for _ in range(n_splits)]
    pf_values = []
    pseudo_values = [[] for _ in range(n_splits)]

    base_model.fit(scaled_DoE, labels)
    prediction_base_model = base_model.predict(scaled_S)
    pf_base_model = np.sum(prediction_base_model <= 0) / nMC
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(scaled_DoE)):
        X_train, X_test = scaled_DoE[train_index], scaled_DoE[test_index]

        y_train, y_test = labels[train_index], labels[test_index]

        # The ith model will be trained on training set where the ith fold is not used for training

        models[i].fit(X_train, y_train)
        predictions[i] = models[i].predict(scaled_S)
        pf = np.sum(predictions[i] <= 0) / nMC
        pf_values.append(pf)

        pseudo_value_i = n_splits * prediction_base_model - (n_splits - 1) * predictions[i]
        pseudo_values[i] = pseudo_value_i
    print(pf_values)
    average_pseudo_value = np.sum(pseudo_values, axis=0) / n_splits

    sigma = np.sum(np.square(pseudo_values - average_pseudo_value), axis=0) / (n_splits * (n_splits - 1))
    learning_values = eff(prediction_base_model, sigma)

    best_point_index = np.argmax(learning_values)
    x_best_point = S[best_point_index]

    label_best_point = np.tanh(performance_function(x_best_point[0], x_best_point[1]))
    labels = np.concatenate((labels, [label_best_point]))
    DoE = np.vstack((DoE, x_best_point))
    scaled_DoE = scaler.transform(DoE)

    delta_pf = np.max(np.abs(pf_base_model - pf_values))
    stopping_criterion = max(learning_values) <= 0.001
    conv_threshold = 0.1
    if (stopping_criterion):
        print("here")
        cov_pf = np.sqrt(1 - pf_base_model) / (np.sqrt(pf_base_model * nMC))
        if (cov_pf <= conv_threshold):
            # Coefficient of variation is acceptable, stop AK-MCS
            print("New ANN finished. Probability of failure: {:.8e}".format(pf_base_model))
            print("Coefficient of variation: {:.2%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
        else:
            new_x1 = np.random.normal(0, 1, size=nMC)
            new_x2 = np.random.normal(0, 1, size=nMC)
            new_points = np.column_stack((new_x1, new_x2))

            S = np.vstack((S, new_points))
            scaled_S = scaler.transform(S)
            nMC = len(S)


    else:
        print("pf", pf_base_model)
        print("stop", max(learning_values))
        iter += 1







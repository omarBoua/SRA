import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold



#Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def performance_function( x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        global function_calls
        function_calls += 1
        return min(term1, term2, term3, term4)

function_calls = 0
def min_distances_from_doe_vectorized(S, D):
    # Convert S and D into numpy arrays
    S_np = np.array(S)
    D_np = np.array(D)
    
    # Expand dimensions to broadcast the subtraction operation
    diffs = S_np[:, np.newaxis] - D_np
    
    # Calculate squared distances
    squared_distances = np.sum(diffs**2, axis=-1)
    
    # Find the minimal squared distance along the last dimension
    min_squared_distances = np.min(squared_distances, axis=-1)
    
    # Return the square root to get the Euclidean distances
    return np.sqrt(min_squared_distances)

all_values = []
all_f = []
for _ in range(25):
    nMC = 500000
    x1 = np.random.normal(0, 1, size=nMC)
    x2 = np.random.normal(0, 1, size=nMC)
    S = np.column_stack((x1, x2))


    #2. create the initial design of experimental
    n_EDini = 12
    selected_indices = np.random.choice(len(S), n_EDini, replace=False)

    DoE = np.array(S[selected_indices])

    initial_design = np.array(DoE)
    labels = np.zeros(n_EDini)
    for i in range(n_EDini):
        labels[i] = performance_function(initial_design[i, 0],
                      initial_design[i,1])  # Evaluate performance function


    scaler = StandardScaler()
    DoE = initial_design
    scaled_DoE = scaler.fit_transform(DoE)




    # Create a k-fold cross validator
    n_splits = 5

    scaled_S = scaler.fit_transform(S)

    models = []
    kernel = C(1.0, (1e-2, 1e2)) * RBF([1,1], (1e-3, 1e3))  # Decreased lower bound from 1e-2 to 1e-3

    for _ in range(n_splits):
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=300)
        models.append(model)

    base_model = GaussianProcessRegressor(kernel=kernel , n_restarts_optimizer=300)
    iter = 0
    kf = KFold(n_splits=n_splits  )

    i=0
    while True :

        predictions =  [[] for _ in range(n_splits)]
        pf_values = []

        base_model.fit(scaled_DoE,labels)
        prediction_base_model,variance = base_model.predict(scaled_S,return_std=True)
        pf_base_model =   np.sum(prediction_base_model <= 0) / nMC
        # Loop through each fold
        for i, (train_index, test_index) in enumerate(kf.split(scaled_DoE)):
            X_train, X_test = scaled_DoE[train_index], scaled_DoE[test_index]

            y_train, y_test = labels[train_index], labels[test_index]

            # The ith model will be trained on training set where the ith fold is not used for training


            models[i].fit(X_train,y_train)
            predictions[i] = models[i].predict(scaled_S)
            pf = np.sum(predictions[i]  <= 0) / nMC
            pf_values.append(pf)

        print(pf_values)
        d_min = min_distances_from_doe_vectorized(scaled_S,scaled_DoE)

        learning_values = np.abs(prediction_base_model) / d_min

        best_point_index = np.argmin(learning_values)
        x_best_point = scaled_S[best_point_index]


        label_best_point = (performance_function(x_best_point[0], x_best_point[1]))
        labels = np.concatenate((labels, [label_best_point]))
        DoE = np.vstack((DoE, x_best_point))
        scaled_DoE = scaler.transform(DoE)


        delta_pf = np.max(np.abs(pf_base_model - pf_values))
        stopping_criterion = delta_pf / pf_base_model
        conv_threshold = 0.01
        if(stopping_criterion <= conv_threshold):
            print("here")
            cov_pf = np.sqrt(1 - pf_base_model) / (np.sqrt(pf_base_model* nMC) )
            if (cov_pf <= 0.02):
                # Coefficient of variation is acceptable, stop AK-MCS
                print("New kriging finished. Probability of failure: {:.2e}".format(pf_base_model))
                print("Coefficient of variation: {:.2%}".format(cov_pf))
                print("Number of calls to the performance function", function_calls)
                all_values.append(pf_base_model)
                all_f.append(function_calls)
                function_calls = 0
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
            print("stop" , stopping_criterion)
            iter += 1



if len(all_values) == 0:
    print("The list is empty.")
    mean = 0
    variance = 0
else:
    mean = sum(all_values) / len(all_values)
    variance = sum((x - mean) ** 2 for x in all_values) / len(all_values)
mean_f = sum(all_f)/len(all_f)
log_file_path = "statistics_log_kriging_distance.txt"
# P_F value
P_F = 4.45e-3

# Calculate the RMSE
rmse = math.sqrt(sum((x - P_F) ** 2 for x in all_values) / len(all_values))
std_dev = math.sqrt(variance)

# Calculate the coefficient of variation
coef_of_variation = std_dev / mean
standardized_rmse = rmse / P_F


# Write logs to the file
with open(log_file_path, 'w') as log_file:
    log_file.write(f"List: {all_values}\n")
    log_file.write(f"Mean: {mean}\n")
    log_file.write(f"Variance: {variance}\n")
    log_file.write(f"Coefficient of Variation: {coef_of_variation}\n")
    log_file.write(f"RMSE with respect to P_F: {rmse}\n")
    log_file.write(f"Standardized RMSE: {standardized_rmse}\n")
    log_file.write(f"Mean function calls: {mean_f}\n")





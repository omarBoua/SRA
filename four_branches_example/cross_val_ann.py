import numpy as np 

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import math
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


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
all_values = []
all_f = []
for _ in range(25):

    function_calls = 0

    nMC = 1000000
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
        labels[i] = np.tanh(labels[i]) #smoothing the labels


    scaler = StandardScaler()
    DoE = initial_design
    scaled_DoE = scaler.fit_transform(DoE)




    # Create a k-fold cross validator
    n_splits = 5

    scaled_S = scaler.fit_transform(S)

    models = []
    for _ in range(n_splits):
        model = MLPRegressor(hidden_layer_sizes=(40), max_iter = 100000, activation='tanh', solver='lbfgs',early_stopping=True)
        models.append(model)

    base_model = MLPRegressor(hidden_layer_sizes=(40), max_iter = 100000, activation='tanh', solver='lbfgs',early_stopping=True)
    iter = 0
    kf = KFold(n_splits=n_splits  )

    i=0
    while True :

        predictions =  [[] for _ in range(n_splits)]
        pf_values = []
        pseudo_values = [[] for _ in range(n_splits)]

        base_model.fit(scaled_DoE,labels)
        prediction_base_model = base_model.predict(scaled_S)
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

            pseudo_value_i = n_splits * prediction_base_model - (n_splits - 1) * predictions[i]
            pseudo_values[i]= pseudo_value_i
        print(pf_values)
        average_pseudo_value = np.sum(pseudo_values, axis= 0)/n_splits

        sigma =  np.sum(np.square(pseudo_values - average_pseudo_value), axis = 0) / (n_splits *(n_splits -1))
        learning_values = np.abs(prediction_base_model) / sigma

        best_point_index = np.argmin(learning_values)
        x_best_point = S[best_point_index]


        label_best_point = np.tanh(performance_function(x_best_point[0], x_best_point[1]))
        labels = np.concatenate((labels, [label_best_point]))
        DoE = np.vstack((DoE, x_best_point))
        scaled_DoE = scaler.transform(DoE)


        delta_pf = np.max(np.abs(pf_base_model - pf_values))
        stopping_criterion = delta_pf / pf_base_model
        conv_threshold = 0.02
        if(stopping_criterion <= conv_threshold):
            cov_pf = np.sqrt(1 - pf_base_model) / (np.sqrt(pf_base_model* nMC) )
            if (cov_pf <= conv_threshold):
                # Coefficient of variation is acceptable, stop AK-MCS
                print("New ANN finished. Probability of failure: {:.2e}".format(pf_base_model))
                print("Coefficient of variation: {:.2%}".format(cov_pf))
                print("Number of calls to the performance function", function_calls)
                all_values.append(pf_base_model)
                all_f.append(function_calls)
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
log_file_path = "statistics_corss_val_ann.txt"
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








        

import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


#Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def performance_function(X):
    global function_calls
    function_calls += 1
    n = len(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)

function_calls = 0
nMC = 300000 # Number of instances to generate
n = 10  # Number of parameters

mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n))







# Stage 2: Definition of initial design of experiments (DoE)
N1 = 12
n_EDini = N1 



DoE = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(N1, n))

scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)

labels = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    labels[i] = performance_function(DoE[i])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])



# Create a k-fold cross validator
n_splits = 5

scaled_S = scaler.transform(S)

models = []
for _ in range(n_splits):
    model = MLPRegressor(hidden_layer_sizes=(10), max_iter = 100000, activation='tanh', solver='lbfgs',early_stopping=True)
    models.append(model)

base_model = MLPRegressor(hidden_layer_sizes=(10), max_iter = 100000, activation='tanh', solver='lbfgs',early_stopping=True)
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
    

    label_best_point = np.tanh(performance_function(x_best_point))
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
            break
        else: 
            new_points = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n)) 
            S = np.vstack((S, new_points))
            scaled_S = scaler.transform(S)
            nMC = len(S)


    else: 
        print("pf", pf_base_model)
        print("stop" , stopping_criterion)
        iter += 1

    



        

import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import SS_surrogate as ss
warnings.filterwarnings("ignore")


#Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def gcal1(X, mu ,sdev):
        k = 6
        term1 = 3 + 0.1 * (X[0] - X[1])**2 - (X[0] + X[1])/(np.sqrt(2))
        term2 = 3 + 0.1 * (X[0] - X[1])**2 + (X[0] + X[1])/(np.sqrt(2))
        term3 = (X[0] - X[1]) + k / (2**0.5)
        term4 = (X[0] - X[1]) + k / (2**0.5)
        global function_calls
        function_calls += 1
        return min(term1, term2, term3, term4)
function_calls = 0

nMC = 500000
dim  = 2
S = np.random.normal(0, 1, size=(nMC,dim))
mu = [0,0]
sdev = [1,1]
#2. create the initial design of experimental 
n_EDini = 12
selected_indices = np.random.choice(len(S), n_EDini, replace=False)

DoE = np.array(S[selected_indices])

initial_design = np.array(DoE)
labels = np.zeros(n_EDini) 
for i in range(n_EDini):
    labels[i] = gcal1(initial_design[i], mu ,sdev)  # Evaluate performance function
    labels[i] = (labels[i]) #smoothing the labels 


DoE = initial_design




# Create a k-fold cross validator
n_splits = 5


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

    base_model.fit(DoE,labels)
    prediction_base_model = base_model.predict(S)  
    pf_base_model =   np.sum(prediction_base_model <= 0) / nMC
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(DoE)):
        X_train, X_test = DoE[train_index], DoE[test_index]
       
        y_train, y_test = labels[train_index], labels[test_index]

        # The ith model will be trained on training set where the ith fold is not used for training
        

        models[i].fit(X_train,y_train)
        predictions[i] = models[i].predict(S)
        pf,cov_ss = ss.SS(base_model,dim)
        pf_values.append(pf)

        pseudo_value_i = n_splits * prediction_base_model - (n_splits - 1) * predictions[i]
        pseudo_values[i]= pseudo_value_i
    print(pf_values)
    average_pseudo_value = np.sum(pseudo_values, axis= 0)/n_splits

    sigma =  np.sum(np.square(pseudo_values - average_pseudo_value), axis = 0) / (n_splits *(n_splits -1))  
    learning_values = np.abs(prediction_base_model) / sigma

    best_point_index = np.argmin(learning_values)
    x_best_point = S[best_point_index]
    

    label_best_point = (gcal1(x_best_point,mu,sdev))
    labels = np.concatenate((labels, [label_best_point]))
    DoE = np.vstack((DoE, x_best_point))


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
            new_x1 = np.random.normal(0, 1, size=nMC)
            new_x2 = np.random.normal(0, 1, size=nMC)
            new_points = np.column_stack((new_x1, new_x2)) 

            S = np.vstack((S, new_points))
            nMC = len(S)


    else: 
        print("pf", pf_base_model)
        print("stop" , stopping_criterion)
        iter += 1

    



        

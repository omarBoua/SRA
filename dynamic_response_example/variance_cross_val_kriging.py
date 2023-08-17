import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold



#Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def performance_function(c1, c2, m, r, t1, F1):
    w0 = np.sqrt((c1 * c2)/m)
    k = 3 
    global function_calls 
    function_calls += 1
    return k * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))
function_calls = 0

nMC = 70000
m = np.random.normal(1, 0.05, size=nMC)
c1 = np.random.normal(1, 0.1, size=nMC)
c2 = np.random.normal(0.1, 0.01, size=nMC)
r = np.random.normal(0.5, 0.05, size=nMC)
F1 = np.random.normal(1, 0.2, size=nMC)
t1 = np.random.normal(1, 0.2, size=nMC) 
S = np.column_stack((c1, c2, m, r, t1, F1))

#2. create the initial design of experimental 
n_EDini = 12
selected_indices = np.random.choice(len(S), n_EDini, replace=False)

DoE = np.array(S[selected_indices])

initial_design = np.array(DoE)
labels = np.zeros(n_EDini) 
for i in range(n_EDini):
    labels[i] = performance_function(DoE[i, 0], DoE[i, 1],DoE[i,2], DoE[i,3], DoE[i,4], DoE[i,5])  # Evaluate performance function


scaler = StandardScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)




# Create a k-fold cross validator
n_splits = 5

scaled_S = scaler.fit_transform(S)

models = []
kernel = C(1.0, (1e-3, 1e3)) * RBF(np.repeat([0.5],6), (1e-2, 1e3))  # Decreased lower bound from 1e-2 to 1e-3

for _ in range(n_splits):
    model = GaussianProcessRegressor(kernel= kernel, n_restarts_optimizer= 30)
    models.append(model)

base_model = GaussianProcessRegressor(kernel= kernel, n_restarts_optimizer= 30)
iter = 0
kf = KFold(n_splits=n_splits  )

i=0
while True :

    predictions =  [[] for _ in range(n_splits)]
    pf_values = []
    pseudo_values = [[] for _ in range(n_splits)]

    base_model.fit(scaled_DoE,labels)
    prediction_base_model = base_model.predict(scaled_S)  
    pf_base_model =   np.sum(prediction_base_model >= 0) / nMC
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(scaled_DoE)):
        X_train, X_test = scaled_DoE[train_index], scaled_DoE[test_index]
       
        y_train, y_test = labels[train_index], labels[test_index]

        # The ith model will be trained on training set where the ith fold is not used for training
        

        models[i].fit(X_train,y_train)
        predictions[i] = models[i].predict(scaled_S)
        pf = np.sum(predictions[i]  >= 0) / nMC
        pf_values.append(pf)

        pseudo_value_i = n_splits * prediction_base_model - (n_splits - 1) * predictions[i]
        pseudo_values[i]= pseudo_value_i
    print(pf_values)
    average_pseudo_value = np.sum(pseudo_values, axis= 0)/n_splits

    sigma =  np.sum(np.square(pseudo_values - average_pseudo_value), axis = 0) / (n_splits *(n_splits -1))  
    learning_values = np.abs(prediction_base_model) / sigma

    best_point_index = np.argmin(learning_values)
    x_best_point = S[best_point_index]
    

    label_best_point = performance_function(x_best_point[0], x_best_point[1],x_best_point[2], x_best_point[3], x_best_point[4], x_best_point[5])
    labels = np.concatenate((labels, [label_best_point]))
    DoE = np.vstack((DoE, x_best_point))
    scaled_DoE = scaler.transform(DoE)


    delta_pf = np.max(np.abs(pf_base_model - pf_values))
    stopping_criterion = delta_pf / pf_base_model
    conv_threshold = 0.02
    if(stopping_criterion <= conv_threshold):
        print("here")
        cov_pf = np.sqrt(1 - pf_base_model) / (np.sqrt(pf_base_model* nMC) )
        if (cov_pf <= 0.05):
            # Coefficient of variation is acceptable, stop AK-MCS
            print("New ANN finished. Probability of failure: {:.2e}".format(pf_base_model))
            print("Coefficient of variation: {:.2%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
        else: 
            new_m = np.random.normal(1, 0.05, size=nMC)
            new_c1 = np.random.normal(1, 0.1, size=nMC)
            new_c2 = np.random.normal(0.1, 0.01, size=nMC)
            new_r = np.random.normal(0.5, 0.05, size=nMC)
            new_F1 = np.random.normal(1, 0.2, size=nMC)
            new_t1 = np.random.normal(1, 0.2, size=nMC) 
            new_points = np.column_stack((new_c1, new_c2, new_m, new_r, new_t1, new_F1)) 
            S = np.vstack((S, new_points))
            scaled_S = scaler.transform(S)
            nMC = len(S)

    else: 
        print("pf", pf_base_model)
        print("stop" , stopping_criterion)
        iter += 1

    



        

import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

def calculate_u_vectorized(samples, models):
    B = len(models)
    predictions = np.zeros((B, samples.shape[0]))
    for i, model in enumerate(models):
        predictions[i] = model.predict(scaler.transform(samples))
    bf = np.sum(predictions <= 0, axis=0)
    bs = np.sum(predictions > 0, axis=0)
    return np.abs(bf - bs) / B





#Performance function with two inputs six input parameters. Set k to 1.5 for lower probability
def g(c1, c2, m, r, t1, F1):
    global function_calls
    w0 = np.sqrt((c1 * c2)/m)
    function_calls += 1
    k = 3
    return k * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))

#needed for plotting
pf_hat_values = []  # List to store pf_hat values at each iteration
function_calls_values = []
pf_max_values = []
pf_min_values = []
cov_pf_values = []
function_calls = 0


#1. generate Monte Carlo Population
nMC = 70000
m = np.random.normal(1, 0.05, size=nMC)
c1 = np.random.normal(1, 0.1, size=nMC)
c2 = np.random.normal(0.1, 0.01, size=nMC)
r = np.random.normal(0.5, 0.05, size=nMC)
F1 = np.random.normal(1, 0.2, size=nMC)
t1 = np.random.normal(1, 0.2, size=nMC) 
S = np.column_stack((c1, c2, m, r, t1, F1))

#parameters for adaptive design
n_epochs_add = 5
n_clusters = 3
n_add = 3 

# Perform k-means clustering on the population S
kmeans = KMeans(n_clusters=n_clusters, max_iter=5, random_state=0)
cluster_labels = kmeans.fit_predict(S)
cluster_labels = kmeans.labels_





#2. create the initial design of experimental 
n_EDini = 50
n_epochs = 200
# Step 1: Find the sample closest to the mean
mean_population = np.mean(S, axis=0)
distances_to_mean = cdist([mean_population], S)
closest_sample_index = np.argmin(distances_to_mean)
initial_design = [S[closest_sample_index]]




# Step 2: Select the remaining points iteratively
for _ in range(n_EDini - 1):
    distances_to_design = cdist(initial_design, S)
    farthest_sample_index = np.argmax(np.min(distances_to_design, axis=0))
    initial_design.append(S[farthest_sample_index])

initial_design = np.array(initial_design)    
labels = np.zeros(n_EDini) 
for i in range(n_EDini):
    labels[i] = g(initial_design[i, 0], 
                  initial_design[i,1],
                    initial_design[i,2],
                      initial_design[i,3],
                        initial_design[i,4],
                          initial_design[i,5])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])      #smoothing the labels

#scaling the input
scaler = MinMaxScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)


 
#3. train the B neural network
B = 50  #number of neural networks
iter = 0 
hidden_layers = np.repeat([2,3,4,5,6,7,8,9,10,11], 5)

last_five_iter_scores = np.zeros(5)
models = [] 
for j in hidden_layers:
    hidden_layer_sizes = (j,j)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', max_iter = n_epochs ,solver = 'lbfgs')
    models.append(model)

scaled_S = scaler.fit_transform(S)
    
while(1):
    validation_errors = []
    pf_values = []
    X_train, X_test, y_train, y_test = train_test_split(scaled_DoE, labels)
    
    #4. update the hyperparameters of the ANNs and get validation errors
    for model in models:

        params = model.get_params()
        index_model = models.index(model)

        new_hidden_layer_size = hidden_layers[index_model]

        params['hidden_layer_sizes'] = (new_hidden_layer_size,new_hidden_layer_size)

        model.set_params(**params)
        model.set_params(max_iter = n_epochs)
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        validation_errors.append(mean_squared_error(y_test, y_test_pred ))

        y_ann = model.predict(scaled_S)
        pf = np.sum(y_ann >= 0) / nMC
        pf_values.append(pf)
    
    #current estimation of probability of failure
    pf_hat = np.mean(pf_values) 
      
    #needed for plotting
    pf_hat_values.append(pf_hat)
    function_calls_values.append(function_calls)
    
   
    cov_pf_iter = np.std(pf_values) / pf_hat
    eps_pf = 0.05 #threshold for convergence
    if(cov_pf_iter <= eps_pf  ):        #check for convergence
        print("Active learning finished. Probability of failure: {:.2e}".format(pf_hat))
        print("Coefficient of variation: {:.2%}".format(cov_pf_iter))
        print("Number of calls to the performance function", function_calls)
        #8. Stop
        break

    #5. find the best next points
    best_points = []
    for cluster_id in range(n_add):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_samples = S[cluster_indices]
        ufbr_values = calculate_u_vectorized(cluster_samples, models )
        best_index = np.argmin(ufbr_values)
        best_point = cluster_samples[best_index]
        best_points.append(best_point)
    best_points = np.array(best_points)
    labels_best_points = np.zeros(n_add)
    for i in range(n_add):
        labels_best_points[i] = g(best_points[i][0]
                                  , best_points[i][1]
                                  ,best_points[i,2]
                                  , best_points[i,3]
                                  , best_points[i,4]
                                  , best_points[i,5])
        labels_best_points[i] = np.tanh(labels_best_points[i])

    #6. update the design of experiment
    labels = np.append(labels, labels_best_points)
    DoE = np.vstack((DoE, best_points))
    scaled_DoE = scaler.transform(DoE)
    n_ED = len(DoE)
    n_epochs =  n_epochs_add * (n_ED - n_EDini) + n_EDini

    #7. define the neural networks to be updated
    alpha = 1.5
    perf_limit = np.min(validation_errors) + alpha * np.std(validation_errors)
    num_layers_to_update = len(np.where(validation_errors > perf_limit)[0])
    num_layers_to_update = min(num_layers_to_update , B//2)
    num_layers_to_update = max(1,num_layers_to_update)

    # Find the indices of the worst neural networks
    worst_model_indices = np.argsort(validation_errors)[-num_layers_to_update:]
    best_model_indices = np.argsort(validation_errors)[:num_layers_to_update]

    # Update the hidden layers of the worst neural networks
    updated_hidden_layers = hidden_layers.copy()
    for index in worst_model_indices:
        if updated_hidden_layers[index] < 13:
            updated_hidden_layers[index] += 1
        else:
            k = np.where(worst_model_indices == index)[0][0]  # Get the index of the current model
            replacement_model_index = best_model_indices[k]  # Get the index of the k-th best model
            models[index] = models[replacement_model_index] 
    hidden_layers = updated_hidden_layers

    #print progress
    print("iter ", iter, ": Probability of failure", pf_hat)
    iter += 1  
   


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

 

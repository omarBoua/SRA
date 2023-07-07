import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
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



pf_hat_values = []  # List to store pf_hat values at each iteration
function_calls_values = []
pf_max_values = []
pf_min_values = []
cov_pf_values = []
function_calls = 0

#limit state function with two inputs x1 and x2

def g(X):
    n = len(X)
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)


#1. generate nMC 
nMC = 300000 # Number of instances to generate
n = 40  # Number of parameters

mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n))


n_epochs_add = 5
n_clusters = 3
n_add = 3 

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, max_iter=5, random_state=0)
cluster_labels = kmeans.fit_predict(S)
cluster_labels = kmeans.labels_

# Access the cluster assignments




#2. initial experimental design
n_EDini = 50
n_epochs = n_EDini
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

labels = np.zeros(n_EDini) 
initial_design = np.array(initial_design)


for i in range(n_EDini):
    labels[i] = g(initial_design[i])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])


scaler = StandardScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)


 
#3. train the B neural network
B = 50  #number of neural networks
iter = 0 
hidden_layers = np.append(np.repeat([2,3,4,5,6,7,8,9,10], 5),[10,10,10,10,10])

models = [] 
for j in hidden_layers:
    hidden_layer_sizes = (j,j,j,j)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', max_iter = n_epochs ,solver = 'lbfgs')
    models.append(model)
    
while(1):
    
    pf_values = []
    validation_errors = []

    for model in models:
        params = model.get_params()
        index_model = models.index(model)

        new_hidden_layer_size = hidden_layers[index_model]

        params['hidden_layer_sizes'] = (new_hidden_layer_size,new_hidden_layer_size,new_hidden_layer_size,new_hidden_layer_size)

        model.set_params(**params)
        model.set_params(max_iter = n_epochs)
        X_train, X_test, y_train, y_test = train_test_split(scaled_DoE, labels)
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)
        validation_errors.append(mean_squared_error(y_test, y_test_pred ))

        scaled_S = scaler.fit_transform(S)
        y_ann = model.predict(scaled_S)
        pf = np.sum(y_ann <= 0) / nMC
        pf_values.append(pf)
        

    eps_pf = 0.05

    pf_hat = np.mean(pf_values)
    pf_max = np.max(pf_values)
    pf_min = np.min(pf_values)
    print("iter ", iter, ":", pf_hat)
    pf_hat_values.append(pf_hat)
    pf_max_values.append(pf_max)
    pf_min_values.append(pf_min)
    function_calls_values.append(function_calls)
    
   
    cov_pf = np.std(pf_values)/ pf_hat
    print("cov: ", cov_pf)
    if(cov_pf <= eps_pf):
        break
  
        



    
    

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
        labels_best_points[i] = g(best_points[i])
        labels_best_points[i] = np.tanh(labels_best_points[i])


    labels = np.append(labels, labels_best_points)
    DoE = np.vstack((DoE, best_points))
    scaled_DoE = scaler.transform(DoE)
    n_ED = len(DoE)
    n_epochs =  n_epochs_add * (n_ED - n_EDini) + n_EDini

   
    alpha = 1.5
    perf_limit = np.min(validation_errors) + alpha * np.std(validation_errors)
    num_layers_to_update = len(np.where(validation_errors > perf_limit)[0])
    num_layers_to_update = min(num_layers_to_update , B//2)
    num_layers_to_update = max(1,num_layers_to_update)



    updated_hidden_layers = hidden_layers.copy()

    # Find the indices of the worst neural networks
    worst_model_indices = np.argsort(validation_errors)[-num_layers_to_update:]
    best_model_indices = np.argsort(validation_errors)[:num_layers_to_update]
# Update the hidden layers of the worst neural networks
    for index in worst_model_indices:
        if updated_hidden_layers[index] < 10:
            updated_hidden_layers[index] += 1
        else:

            k = np.where(worst_model_indices == index)[0][0]  # Get the index of the current model
            replacement_model_index = best_model_indices[k]  # Get the index of the k-th best model
            models[index] = models[replacement_model_index] 

    hidden_layers = updated_hidden_layers
    
    iter += 1  



#uncomment for plotting probabilities against number of lsf calls
 # Plotting pf_hat values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'b-')
plt.xlabel('function_calls')
plt.ylabel('pf_hat')
plt.title('Convergence Plot')

# Indicate the last point
last_point_calls = function_calls_values[-1]
last_point_pf_hat = pf_hat_values[-1]
plt.plot(last_point_calls, last_point_pf_hat, 'ro')
plt.annotate(f'({last_point_calls}, {last_point_pf_hat:.4e})',
             xy=(last_point_calls, last_point_pf_hat),
             xytext=(last_point_calls + 2, last_point_pf_hat+last_point_pf_hat ),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Display the number of iterations
plt.text(0.95, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show() 


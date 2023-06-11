import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
def calculate_u(sample, models):
    bf = 0
    bs = 0 
    B = len(models)
    for model in models:
        
        y_ann = model.predict(scaler.transform([sample]))
        if y_ann <= 0:
            bf += 1
        else: 
            bs += 1 
    return np.abs(bf - bs) / B  

pf_hat_values = []  # List to store pf_hat values at each iteration
function_calls_values = []
function_calls = 0

#limit state function with two inputs x1 and x2
def LSF(x1, x2):
    global function_calls
    k = 6
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    function_calls += 1
    return min(term1, term2, term3, term4)
#1. generate nMC 
nMC = 5000
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))

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
    labels[i] = LSF(initial_design[i, 0], initial_design[i,1])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])


scaler = StandardScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)
#kernel = ConstantKernel(1.0) * RBF(1.0)


 
#3. train the B neural network
B = 40  #number of neural networks
iter = 0 
hidden_layers = np.repeat([2, 3, 4, 5], 10)
while(1):
    losses = []
    models = [] 

    for j in hidden_layers:
        hidden_layer_sizes = (j,)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', max_iter = n_epochs ,solver = 'lbfgs')
        model.fit(scaled_DoE,labels)
        models.append(model)
    
    pf_values = []
    
    for model in models:
        scaled_S = scaler.fit_transform(S)
        y_ann = model.predict(scaled_S)
        pf = np.sum(y_ann <= 0) / nMC
        pf_values.append(pf)
        losses.append(model.loss_)

    worst_model_index = np.argmax(losses)
    best_model_index = np.argmin(losses)
    eps_pf = 0.5

    pf_hat = np.mean(pf_values)
    pf_max = np.max(pf_values)
    pf_min = np.min(pf_values)
    relative_change = (pf_max - pf_min) / pf_hat
    print("iter ", iter, ":", pf_hat)
    print(relative_change)
    pf_hat_values.append(pf_hat)
    function_calls_values.append(function_calls)
    if(relative_change <= eps_pf):
        break
    
    if(iter == 30):
        break
    

    best_points = []
    for cluster_id in range(n_add):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_samples = S[cluster_indices]
        ufbr_values = np.array([calculate_u(sample, models ) for sample in cluster_samples])
        best_index = np.argmin(ufbr_values)
        best_point = cluster_samples[best_index]
        best_points.append(best_point)
    best_points = np.array(best_points)
    labels_best_points = np.zeros(n_add)

    for i in range(n_add):
        labels_best_points[i] = LSF(best_points[i][0], best_points[i][1])
        labels_best_points[i] = np.tanh(labels_best_points[i])
    labels = np.append(labels, labels_best_points)
    DoE = np.vstack((DoE, best_points))
    scaled_DoE = scaler.transform(DoE)
    n_ED = len(DoE)
    n_epochs =  n_epochs_add * (n_ED - n_EDini) + n_EDini

    if(hidden_layers[worst_model_index] < 5):
        hidden_layers[worst_model_index] += 1 
   
    print(function_calls)
   
    iter += 1 
#uncomment for plotting probabilities against number of lsf calls
""" # Plotting pf_hat values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'bo-')
plt.xlabel('function_calls')
plt.ylabel('pf_hat')
plt.title('Convergence Plot')

# Indicate the last point
last_point_calls = function_calls_values[-1]
last_point_pf_hat = pf_hat_values[-1]
plt.plot(last_point_calls, last_point_pf_hat, 'ro')
plt.annotate(f'({last_point_calls}, {last_point_pf_hat})',
             xy=(last_point_calls, last_point_pf_hat),
             xytext=(last_point_calls + 2, last_point_pf_hat+last_point_pf_hat ),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Display the number of iterations
plt.text(0.95, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show() """


#uncomment for plotting design of experiment

x1_vals = np.linspace(-6, 6, 1000)
x2_vals = np.linspace(-6, 6, 1000)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Calculate LSF values for each combination of x1 and x2
Z = np.array([LSF(x1, x2) for x1, x2 in zip(X1.flatten(), X2.flatten())])
Z = Z.reshape(X1.shape)

# Plotting the contour of LSF
plt.contour(X1, X2, Z, levels=[0], colors='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('LSF Contour')

# Plotting the initial points in the design of experiment
plt.scatter(DoE[:, 0], DoE[:, 1], c='blue', s=5, label='Initial Points', marker = 'o')

# Plotting the added points in the final design of experiment
plt.scatter(DoE[n_EDini:, 0], DoE[n_EDini:, 1], c='red',s=5, label='Added Points', marker = 'o')

plt.legend()
plt.show()
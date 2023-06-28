import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
np.random.seed(1350)

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
pf_max_values = []
pf_min_values = []
cov_pf_values = []
function_calls = 0

#limit state function with two inputs x1 and x2
""" def LSF(x1, x2):
    global function_calls
    k = 6
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    function_calls += 1
    return min(term1, term2, term3, term4) """
def g(c1, c2, m, r, t1, F1):
    global function_calls
    print("m: ", m)
    w0 = np.sqrt((c1 * c2)/m)
    function_calls += 1

    return 3 * r - np.abs(2 * F1 * np.sin(w0*t1/2)/ (m*w0**2))


#1. generate nMC 
nMC = 5000
""" 
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))
 """

m = np.random.normal(1, 0.05, size=nMC)
c1 = np.random.normal(1, 0.1, size=nMC)
c2 = np.random.normal(0.1, 0.01, size=nMC)
r = np.random.normal(0.5, 0.05, size=nMC)
F1 = np.random.normal(1, 0.2, size=nMC)
t1 = np.random.normal(1, 0.2, size=nMC) 

S = np.column_stack((c1, c2, m, r, t1, F1))


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
    labels[i] = g(initial_design[i, 0], 
                  initial_design[i,1],
                    initial_design[i,2],
                      initial_design[i,3],
                        initial_design[i,4],
                          initial_design[i,5])  # Evaluate performance function
    labels[i] = np.tanh(labels[i])


scaler = StandardScaler()
DoE = initial_design
scaled_DoE = scaler.fit_transform(DoE)
#kernel = ConstantKernel(1.0) * RBF(1.0)


 
#3. train the B neural network
B = 50  #number of neural networks
iter = 0 
hidden_layers = np.repeat([2,3,4,5,6,7,8,9,10,11,12,13], 5)

last_five_iter_scores = np.zeros(5)
while(1):
    losses = []
    models = [] 
    print(hidden_layers)
    for j in hidden_layers:
        hidden_layer_sizes = (j,j)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', max_iter = n_epochs ,solver = 'lbfgs')
        model.fit(scaled_DoE,labels)
        models.append(model)
        
    pf_values = []
    
    for model in models:
        scaled_S = scaler.fit_transform(S)
        y_ann = model.predict(scaled_S)
        pf = np.sum(y_ann >= 0) / nMC
        pf_values.append(pf)
        losses.append(model.loss_)
        

    worst_model_index = np.argmax(losses)
    eps_pf = 0.1

    pf_hat = np.mean(pf_values)
    pf_max = np.max(pf_values)
    pf_min = np.min(pf_values)
    print("iter ", iter, ":", pf_hat)
    pf_hat_values.append(pf_hat)
    pf_max_values.append(pf_max)
    pf_min_values.append(pf_min)
    function_calls_values.append(function_calls)
    
   
    cov_pf_iter = (pf_max - pf_min) / pf_hat
    last_five_iter_scores[iter % 5] = cov_pf_iter

    if(iter % 5 == 0 and iter > 0 ):
        cov_pf = np.mean(last_five_iter_scores)
        cov_pf_values.append(cov_pf)
        if(cov_pf <= eps_pf):
            break
    #cov_pf = np.sqrt(1 - pf_hat) / (np.sqrt(pf_hat* nMC) )



    
    

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
        labels_best_points[i] = g(best_points[i][0]
                                  , best_points[i][1]
                                  ,best_points[i,2]
                                  , best_points[i,3]
                                  , best_points[i,4]
                                  , best_points[i,5])
        labels_best_points[i] = np.tanh(labels_best_points[i])


    labels = np.append(labels, labels_best_points)
    DoE = np.vstack((DoE, best_points))
    scaled_DoE = scaler.transform(DoE)
    n_ED = len(DoE)
    n_epochs =  n_epochs_add * (n_ED - n_EDini) + n_EDini

    """  if(hidden_layers[worst_model_index] < 5):
        hidden_layers[worst_model_index] += 1 
        
    else:
        for i in range(len(hidden_layers)):
            if(hidden_layers[i]<5):
               
                hidden_layers[i] +=1
                break """
    alpha = 1.5
    num_layers_to_update = min(math.ceil(np.min(losses) + alpha * np.std(losses)), B//2)
    num_layers_to_update = 3
    print(num_layers_to_update)
    updated_hidden_layers = hidden_layers.copy()

    # Find the indices of the worst neural networks
    worst_model_indices = np.argsort(losses)[-num_layers_to_update:]

# Update the hidden layers of the worst neural networks
    for index in worst_model_indices:
        if updated_hidden_layers[index] < 13:
            updated_hidden_layers[index] += 1
        else:
            for i in range(len(updated_hidden_layers)):
                if updated_hidden_layers[i] < 13:
                    updated_hidden_layers[i] += 1
                    break

    hidden_layers = updated_hidden_layers
    #print(function_calls)
    
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

""" 
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


legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=1, label='G = 0'),
    plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=5, label='Initial Points'),
    plt.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=5, label='Added Points')
]


plt.legend(handles=legend_elements)

plt.show() """


# uncomment to plot cov_pf_values
""" multiples_of_5 = [i for i in range(5, iter + 1, 5)]

plt.plot(multiples_of_5, cov_pf_values, 'ro-', label='Ratio of Convergence')


# Plotting pf_mcs as a fixed value

plt.xlabel('Iterations')
plt.ylabel('RoC')
plt.title('Convergence Plot')
plt.legend()
plt.xticks(multiples_of_5)
# Indicate the last point


# Display the number of iterations
plt.text(0.05, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='left',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show() """
#uncomment for plotting pf_max, pf_min, pf_hat and pf_mcs
""" 
# Plotting pf_hat, pf_max, and pf_min values vs. function_calls
plt.plot(function_calls_values, pf_hat_values, 'r-', label='pf_hat')
plt.plot(function_calls_values, pf_max_values, 'r--', label='pf_max')
plt.plot(function_calls_values, pf_min_values, 'r--', label='pf_min')

# Plotting pf_mcs as a fixed value
pf_mcs = 0.004447
plt.axhline(y=pf_mcs, color='purple', linestyle=':', label='pf_mcs')

plt.xlabel('Function Calls')
plt.ylabel('pf')
plt.title('Convergence Plot')
plt.legend()

# Indicate the last point
last_point_calls = function_calls_values[-1]
last_point_pf_hat = pf_hat_values[-1]
plt.plot(last_point_calls, last_point_pf_hat, 'ro')
plt.annotate(f'({last_point_calls}, {last_point_pf_hat:.4e})',
             xy=(last_point_calls, last_point_pf_hat),
             xytext=(last_point_calls , last_point_pf_hat + last_point_pf_hat ),
             arrowprops=dict(facecolor='black', arrowstyle='->'))


# Display the number of iterations
plt.text(0.05, 0.95, f'Iterations until convergence: {iter}',
         verticalalignment='top', horizontalalignment='left',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show()
 """

#uncomment for clustering plot
"""  # Create a scatter plot of the data points with colors based on their cluster labels
plt.scatter(S[:, 0], S[:, 1], c=cluster_labels)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
 """
import numpy as np 
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


#limit state function with two inputs x1 and x2
def LSF(x1, x2):
    k = 7
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)
#1. generate nMC 
nMC = 110
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))

labels = np.zeros(nMC)
for i in range(nMC):
    labels[i] = LSF(S[i, 0], S[i,1])  # Evaluate performance function
 

scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(S)


model = MLPRegressor(hidden_layer_sizes=(5,), activation='tanh',solver = 'lbfgs',  max_iter = 100000)
model.fit(S,labels)

nMC = 100000
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))

y_ann = model.predict(S)

pf = np.sum(y_ann<= 0) / nMC 

print(pf)
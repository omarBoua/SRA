
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from plot_script.NN_classifier_class import PerformanceTrainer
# Create an instance of the PerformanceTrainer class
train_set_sizes = [10,50,  75, 100, 500, 1000, 10000 ]
# Call the train_and_get_pf_values() function to train the model and get pf_values
pf_values_NNC = []
for i in train_set_sizes:
    trainer = PerformanceTrainer(N1 = i)
    pf_values_NNC.append(np.mean(trainer.train_and_get_pf_values(20)) ) 



# Create a boxplot of the pf_values

# Create boxplots for NNC and MCS
mean_NNC = np.mean(pf_values_NNC)
std_NNC = np.std(pf_values_NNC)


mean = mean_NNC
std = std_NNC




plt.plot(np.arange(len(train_set_sizes)), pf_values_NNC, marker="o")
# Set x-axis tick labels
plt.xticks(np.arange(len(train_set_sizes)), train_set_sizes)
# Set x-axis tick labels
# Add mean and standard deviation text

# Set labels and title


plt.xlabel('test set size')

plt.ylabel('PF Value')
plt.title('Probability of failure against training set size')
plt.show()





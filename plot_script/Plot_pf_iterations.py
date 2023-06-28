
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from plot_script.NN_classifier_class import PerformanceTrainer
# Create an instance of the PerformanceTrainer class
trainer = PerformanceTrainer()

# Call the train_and_get_pf_values() function to train the model and get pf_values
pf_values_NNC = trainer.train_and_get_pf_values(20)

# Create a boxplot of the pf_values

# Create boxplots for NNC and MCS
mean_NNC = np.mean(pf_values_NNC)
std_NNC = np.std(pf_values_NNC)


mean = mean_NNC
std = std_NNC


iter = np.arange(1, 21)

plt.text(0.95, 0.95, f'Mean: {mean:.2f}\nStd: {std:.4f}', 
         transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

plt.plot(iter, pf_values_NNC , marker ='o')
# Set x-axis tick labels
# Add mean and standard deviation text

# Set labels and title
plt.xlabel('Iterations')
plt.xticks(range(1, 21))

plt.ylabel('PF Value')
plt.title('Probability of failure per iteration')
plt.show()



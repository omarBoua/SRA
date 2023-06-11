import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from NN_classifier_class import PerformanceTrainer

# Create an instance of the PerformanceTrainer class
train_set_sizes = [10,50,  75, 100, 500, 1000, 10000 ]

# Call the train_and_get_pf_values() function to train the model and get pf_values
pf_values_NNC = []
for i in train_set_sizes:
    trainer = PerformanceTrainer(N1 = i)
    pf_values_NNC.append(trainer.train_and_get_pf_values(20))

# Create a boxplot of the mean pf_values for each population size
plt.boxplot(pf_values_NNC, labels=train_set_sizes)

# Add mean and standard deviation text



# Set labels and title
plt.xlabel("Traint Set Size")
plt.ylabel("PF Value")
plt.title("Probability of Failure against Train Set Size")

# Show the plot
plt.show()
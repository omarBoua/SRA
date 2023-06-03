import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def performance_function(x1,x2):
    return x2 - np.abs(np.tan(x1)) - 1



# Define the neural network
class PerformanceNet(nn.Module):
    def __init__(self):
        super(PerformanceNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        
        x = self.fc1(x) #change it to a new function, sigmoid function or tanh (classification problem ?)
        x = torch.sigmoid(self.fc2(x))
        return x

pf_values = []
for i in range(100):
# Generate training data by randomly choosing x1 and x2
    N1 = 500

    x1_train = np.random.normal(4,2,size = N1)
    x2_train    = np.random.normal(-2, 2, size =N1)

    y_train = np.array([1 if performance_function(x1, x2) < 0 else 0 for x1, x2 in zip(x1_train, x2_train)])
    # Convert training data to PyTorch tensors
    x_t = torch.tensor(np.column_stack((x1_train, x2_train)), dtype=torch.float32)
    y_train = y_train.reshape(-1, 1)

    y_t = torch.tensor(y_train, dtype=torch.float32)

    # Create an instance of the neural network
    model = PerformanceNet()

    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the neural network
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model.forward(x_t)
        loss = criterion(outputs, y_t)
        
        # Backward and optimize
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
    

    N_test = 100000
    x1_test = np.random.normal(4, 2, size=N_test) #more samples
    x2_test = np.random.normal(-2, 2, size=N_test)

    x_test = torch.tensor(np.column_stack((x1_test, x2_test)), dtype=torch.float32)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        y_pred = model(x_test)


    y_pred = y_pred.numpy()



    threshold = 0.5
    y_pred_class = np.where(y_pred > threshold, 1, 0)
    Pf_hat = np.sum(y_pred_class == 0) / N_test


    
    
    pf_values.append(Pf_hat)

iteration_numbers = list(range(1, 101))
# Plot the data
pf_mean = np.mean(pf_values)
pf_std = np.std(pf_values)

# Plot the data
plt.plot(iteration_numbers, pf_values, marker='o')

# Add labels and title
plt.xlabel('Iteration Number')
plt.ylabel('PF Value')
plt.title('PF Values for each Iteration')

# Add mean and standard deviation text
plt.text(0.95, 0.95, f'Mean: {pf_mean:.2f}\nStd: {pf_std:.2f}', 
         transform=plt.gca().transAxes, va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

# Show the plot
plt.show()


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
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,1)
    def forward(self, x):
        
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# Generate training data by randomly choosing x1 and x2
N1 = 500

x1_train = np.random.normal(4,2,size = N1)
x2_train    = np.random.normal(2.5, 2, size =N1)

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
    


N_test = 1000000
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

print(Pf_hat)
    
    
 


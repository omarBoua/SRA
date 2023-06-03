import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

""" def performance_function(x1,x2):
    return 10 - (x1**2 - 5* math.cos(2 * math.pi * x1)) - (x2**2 - 5* math.cos(2 * math.pi * x2))
 """
def performance_function(x1,x2):
    return x2 - np.abs(np.tan(x1)) - 1
# Define the performance function
""" def performance_function(x1, x2):
    k = 7
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4) """
# Generate Monte Carlo samples 



# Generate training data by randomly choosing N1 samples from S
N1 =500

x1_train = np.random.normal(0,1,size = N1)
x2_train = np.random.normal(0,1,size = N1)
y_train = np.array([performance_function(x1, x2) for x1, x2 in zip(x1_train, x2_train)])

# Define the neural network
class PerformanceNet(nn.Module):
    def __init__(self):
        super(PerformanceNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x) #change it to a new function, sigmoid function or tanh (classification problem ?)
        x = self.fc2(x)
        return x

# Convert training data to PyTorch tensors
x_t = torch.tensor(np.column_stack((x1_train, x2_train)), dtype=torch.float32)
y_train = y_train.reshape(-1, 1)

y_t = torch.tensor(y_train, dtype=torch.float32)

# Create an instance of the neural network
model = PerformanceNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model.forward(x_t)
    loss = criterion(outputs, y_t)
    
    # Backward and optimize
    optimizer.zero_grad() # initialize randomly, gaissian initialization 
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

n = 3000

x1_test = np.random.normal(0, 1, size=n)
x2_test = np.random.normal(0, 1, size=n)

x_test = torch.tensor(np.column_stack((x1_test, x2_test)), dtype=torch.float32)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    y_pred = model(x_test)


y_pred = y_pred.numpy()

Pf_hat = np.sum(y_pred > 0) / n

print(Pf_hat)

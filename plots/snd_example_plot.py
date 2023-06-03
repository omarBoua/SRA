import matplotlib.pyplot as plt
import numpy as np

# Define the performance function
def performance_function(x1, x2):
    return x2 - np.abs(np.tan(x1)) - 1

# Generate a grid of points to evaluate the performance function
x1 = np.linspace(-8, 16, 10000)  # Adjust the range as per your requirement
x2 = np.linspace(-8, 16, 10000)  # Adjust the range as per your requirement

u1 = np.random.normal(4, 2, size= 3000)
u2 = np.random.normal(-2, 2, size= 3000)

performance_values = performance_function(u1, u2)


negative_values_indices = np.where(performance_values < 0)
positive_values_indices = np.where(performance_values >= 0)

# Plot the data points with negative performance function values in red
plt.scatter(u1[negative_values_indices], u2[negative_values_indices], color='blue', label='Safe region g(u1,u2) < 0', s=2)

# Plot the data points with non-negative performance function values in blue
plt.scatter(u1[positive_values_indices], u2[positive_values_indices], color='red', label='Unsafe region g(u1,u2) > 0', s=2)

X1, X2 = np.meshgrid(x1, x2)

# Apply conditional statement to exclude values where tangent is not defined
Z = performance_function(X1, X2)

# Create a contour plot of the performance function
plt.contour(X1, X2, Z, levels=0,  colors = 'black')

# Set plot title and labels
plt.title('Data points classified into safe and unsafe regions')
plt.xlabel('u1')
plt.ylabel('u2')

plt.xlim(-4, 12)
plt.ylim(-4, 12)
plt.xticks(np.arange(-4, 13, 4))
plt.yticks(np.arange(-4, 13, 4))
plt.legend()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define the performance function
def performance_function(x1, x2):
    k = 8.5
    term1 = 4 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 4 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return np.minimum(np.minimum(term1, term2), np.minimum(term3, term4))
# Generate a grid of points to evaluate the performance function
x1 = np.linspace(-8, 8, 10000)  # Adjust the range as per your requirement
x2 = np.linspace(-8, 8, 10000)  # Adjust the range as per your requirement



u1 = np.random.normal(0, 1, size= 1000000)
u2 = np.random.normal(0, 1, size= 1000000)

performance_values = performance_function(u1, u2)


negative_values_indices = np.where(performance_values <= 0)
positive_values_indices = np.where(performance_values > 0)

# Plot the data points with negative performance function values in red
plt.scatter(u1[negative_values_indices], u2[negative_values_indices], color='red', label='Unafe region g(u1,u2) < 0', s=2)

# Plot the data points with non-negative performance function values in blue
plt.scatter(u1[positive_values_indices], u2[positive_values_indices], color='blue', label='Safe region g(u1,u2) > 0', s=2)



X1, X2 = np.meshgrid(x1, x2)

# Apply conditional statement to exclude values where tangent is not defined
Z = performance_function(X1, X2)

# Create a contour plot of the performance function
plt.contour(X1, X2, Z, levels=0,  colors = 'black',label = "G = 0")

# Set plot title and labels
plt.title('Data points classified into safe and unsafe regions')
plt.xlabel('u1')
plt.ylabel('u2')
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=1, label='G = 0'),
    plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=5, label='Safe region g(u1,u2) > 0'),
    plt.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=5, label='Unsafe region g(u1,u2) < 0')
]
# Add the x = 0 and y = 0 lines
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')


plt.xticks(np.arange(-8, 9, 2))
plt.yticks(np.arange(-8, 9, 2))
plt.legend(handles=legend_elements)

# Show the plot
plt.show()
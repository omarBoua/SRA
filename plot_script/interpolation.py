import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import numpy as np
import matplotlib.pyplot as plt

# Generate 5 random coordinates for red dots
x_red = np.random.rand(5)
y_red = np.random.rand(5)

# Generate coordinates for the middle blue dot (change values as needed)
x_blue = np.mean(x_red) + np.random.uniform(-0.2, 0.2)
y_blue = np.mean(y_red) + np.random.uniform(-0.2, 0.2)

# Ensure the blue dot remains within the bounds [0, 1]
x_blue = max(0, min(x_blue, 1))
y_blue = max(0, min(y_blue, 1))

# Create the plot
plt.scatter(x_red, y_red, color='red', label='Labeled sample')
plt.scatter(x_blue, y_blue, color='blue', label='Unlabeled sample')

# Draw lines connecting the blue dot to the red dots to visually indicate interpolation
for i in range(5):
    plt.plot([x_red[i], x_blue], [y_red[i], y_blue], color='gray', linestyle='--')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

# Save the plot as a .pgf file
plt.savefig('plot.pgf')

# Show the plot

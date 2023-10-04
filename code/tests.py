import numpy as np

def gaussian(x, y, amplitude, mean_x, mean_y, sigma_x, sigma_y):
    exponent = -((x - mean_x)**2 / (2 * sigma_x**2) + (y - mean_y)**2 / (2 * sigma_y**2))
    return amplitude * np.exp(exponent)

# Define the parameters for the 2D Gaussian
amplitude = 1.0  # Amplitude of the Gaussian
mean_x = 0.0     # Mean (center) along the x-axis
mean_y = 0.0     # Mean (center) along the y-axis
sigma_x = 1.0    # Standard deviation along the x-axis
sigma_y = 1.0    # Standard deviation along the y-axis

# Create a grid of x and y values
x = np.linspace(-5, 5, 100)  # Adjust the range and resolution as needed
y = np.linspace(-5, 5, 100)  # Adjust the range and resolution as needed

# Create a 2D matrix to store the Gaussian data
gaussian_matrix = np.zeros((len(x), len(y)))

# Calculate the Gaussian values and fill the matrix
for i in range(len(x)):
    for j in range(len(y)):
        gaussian_matrix[i, j] = gaussian(x[i], y[j], amplitude, mean_x, mean_y, sigma_x, sigma_y)

# Now, the 'gaussian_matrix' contains the amplitude data of the 2D Gaussian

a = 0
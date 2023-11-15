import GPy
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
X = np.random.rand(30, 1)  # 2D input data

X[:, 0] = [
    0.003036903,
    0.002581726,
    0.00284598,
    0.002073653,
    0.008190593,
    0.000999313,
    0.001533293,
    0.002850198,
    0.002984582,
    0.001302356,
    0.002335469,
    0.002716275,
    0.001956693,
    0.003486448,
    0.00234497,
    0.005525822,
    0.003202159,
    0.003884893,
    0.001856687,
    0.003307555,
    0.008632089,
    0.001588094,
    0.002824852,
    0.005571381,
    0.003106295,
    0.002580903,
    0.001867084,
    0.003346787,
    0.003215854,
    0.001137742
]

Y = 0 * X

Y[:, 0] = [
    0.04,
    0.06,
    0.03,
    0.05,
    0.07,
    0.04,
    0.04,
    0.03,
    0.02,
    0.02,
    0.03,
    0.02,
    0.04,
    0.04,
    0.02,
    0.04,
    0.05,
    0.02,
    0.04,
    0.11,
    0.05,
    0.05,
    0.04,
    0.04,
    0.04,
    0.04,
    0.02,
    0.06,
    0.04,
    0.02
]

# Define the Gaussian Process model
kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)  # Radial Basis Function (RBF) kernel
model = GPy.models.GPRegression(X, Y, kernel)

# Optimize the model hyperparameters
model.optimize_restarts(num_restarts=10, verbose=False)

model.save_model("testmodel")

# Make predictions
X_new = np.zeros((100, 1))
X_new[:, 0] = np.linspace(0, 0.009, 100)
# X_new[:, 1] = np.linspace(0, 1, 100)
Y_pred, _ = model.predict(X_new)

yyy = 9.8125 * X_new + 0.0318 - 0.02

print(Y_pred)


# Plot the results
plt.figure()
plt.scatter(X, Y, label='Data', c='b')
plt.plot(X_new, Y_pred, 'r', label='Gaussian Process Regression')
plt.plot(X_new, yyy, 'g', label='Linear Regression')
# plt.title('Gaussian Process Regression')
plt.xlabel('Variance')
plt.ylabel('Optimal threshold')
plt.legend()
plt.show()

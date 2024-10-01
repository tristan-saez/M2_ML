def ridge_regressor(data):
    pass


"""
import numpy as np
import pandas as pd
from numpy.linalg import inv

LAMBDA = 2    # shrinkage parameter

# Define dataset (X,y)
X = np.array([[0.8,  1.2,  0.5,  -0.7, 1.0],
              [1.0,  0.8,  -0.4, 0.5,  -1.2],
              [-0.5, 0.3,  1.2,  0.9,  -0.1],
              [0.2,  -0.9, -0.7, 1.1,  0.5]])

y = np.array([3.2, 2.5, 1.8, 2.9])

# Scale predictors
X_scale = (X-X.mean(axis=0))/X.std(axis=0)

# RIDGE REGRESSION MODEL - coefficients estimation
# X*X^T + LAMBDA*I
x1 = np.matmul(X_scale.T, X_scale) + LAMBDA*np.identity(5)
# Transpose obtained matrix - (X*X^T + LAMBDA*I)^{-1}
x1_inv = inv(x1)
# ( (X*X^T + LAMBDA*I)^{-1} ) * X^T
x2 = np.matmul(x1_inv, X_scale.T)
# ( ( (X*X^T + LAMBDA*I)^{-1} ) * X^T ) * Y
coef = np.matmul(x2, y)
# Estimated coeficients
print(coef)

# predictions
np.matmul(X_scale, coef)+y.mean()


Ensuite il faut un moyen de tester plusieur LAMBDA jusqu'à trouver le bon.
Idée : tester 0.1, 0.5, 1, 2, 5, 10, 100, 1000 
Puis rechercher meilleur en prenant moitier haute basse plusieurs fois jusqu'à obtenir LAMBDA le plus précis

"""
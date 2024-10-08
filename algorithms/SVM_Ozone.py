import numpy as np
from algorithms.NormalizeOzone import PreProcessingOzone
from cvxopt import matrix, solvers
from itertools import product
# Régressseur SVM
class svm_reg:
    """Régresseur de la machine à vecteurs de support, utilise la programmation quadratique pour résoudre le vecteur de poids et biais."""
    def __init__(self, m_type='linear', C=6, gamma=1):
        self.C = C
        self.gamma = gamma
        self.m_type = m_type
        self.alphas = None
        self.svs_y = None
        self.svs_x = None
        self.w = None
        self.nf = None
        self.b = None
        if m_type == 'linear':
            self.kernel = self.linear_kernal
        elif m_type == 'polynomial':
            self.kernel = self.polynomial_kernel
        elif m_type == 'rbf':
            self.kernel = self.rbf_kernel
        elif m_type == 'sigmoid':
            self.kernel = self.sigmoid_kernel
        else:
            raise ValueError('Invalid kernel type')

    def linear_kernal(self, x1, x2):
        return x1.T @ x2

    def polynomial_kernel(self, x1, x2):
        return (x1.T @ x2 + 1) ** self.gamma

    def rbf_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (self.nf * self.gamma ** 2))

    def sigmoid_kernel(self, x1, x2):
        return np.tanh(self.gamma * x1.T @ x2 + self.gamma)

    def fit(self, X: np.array, Y: np.array):
        """Initialise le modèle et calcule le vecteur de poids et b."""
        n, self.nf = X.shape
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        P = matrix(K)
        q = matrix(-Y)
        A = matrix(np.ones((1, n)), tc='d')
        b = matrix(1.0)

        g = np.diag(np.ones(n))
        g2 = -1 * np.diag(np.ones(n))
        g = np.concatenate((g, g2), axis=0)
        g = matrix(g)

        h = self.C * np.ones(n)
        h2 = np.zeros(n)
        h = np.concatenate((h.T, h2.T), axis=0)
        h = matrix(h)

        solution = solvers.qp(P, q, g, h, A, b)
        alphas = np.ravel(solution['x'])

        self.w = np.zeros(self.nf)
        for alpha, y, x in zip(alphas, Y, X):
            self.w += alpha * y * x

        idxs = alphas > 1e-5
        self.svs_x = X[idxs]
        self.svs_y = Y[idxs]
        self.alphas = alphas[idxs]

        self.b = np.mean([self.svs_y[i] - sum([alpha * y * self.kernel(x, self.svs_x[i])
                                               for alpha, y, x in zip(self.alphas, self.svs_y, self.svs_x)])
                          for i in range(len(self.svs_x))])

    def _predict(self, inpt: np.array):
        if self.m_type == 'linear':
            return self.w @ inpt + self.b
        else:
            return sum([(alpha * y * self.kernel(sv_x, inpt)) for alpha, y, sv_x in zip(self.alphas, self.svs_y, self.svs_x)]) + self.b

    def predict(self, inpt: np.array):
        """Prend un vecteur d'entrée et calcule le résultat scalaire attendu."""
        inpt = np.array(inpt)
        if len(inpt.shape) == 1:
            return self._predict(inpt)
        else:
            res = np.zeros(len(inpt))
            for i in range(len(inpt)):
                res[i] = self._predict(inpt[i])
            return res


def mean_squared_error(y_true, y_pred):
    """Calcule l'erreur quadratique moyenne du modèle."""
    return np.mean((y_true - y_pred) ** 2)

def svm(X_train, X_test, Y_train, Y_test):

    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values
    
    
    model = svm_reg(m_type='rbf', C=0.001, gamma=0.001)
    model.fit(X_train, Y_train)
    
    
    predictions = model.predict(X_test)
    return Y_test, predictions
    
# X_train, X_test, Y_train, Y_test = PreProcessingOzone('data/ozone_complet.txt', ';')
# y_test, predictions=svm(X_train, X_test, Y_train, Y_test)

    
# if __name__ == "__main__":

#     X_train, X_test, Y_train, Y_test = PreProcessingOzone('data/ozone_complet.txt', ';')
#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_test = X_test.values
#     Y_test = Y_test.values

#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],  
#         'gamma': [0.001, 0.01, 0.1, 1, 10, 50, 100],     
#         'm_type': ['linear', 'polynomial', 'rbf', 'sigmoid']  
#     }

#     best_mse = float("inf")
#     best_params = {}

#     for C, gamma, m_type in product(param_grid['C'], param_grid['gamma'], param_grid['m_type']):
#         print(f"Test des paramètres : C={C}, gamma={gamma}, m_type={m_type}")

#         model = svm_reg(m_type=m_type, C=C, gamma=gamma)
#         model.fit(X_train, Y_train)

#         predictions = model.predict(X_test)
#         mse = mean_squared_error(Y_test, predictions)
#         print(f"Mean Squared Error = {round(mse, 4)}")

#         if mse < best_mse:
#             best_mse = mse
#             best_params = {'C': C, 'gamma': gamma, 'm_type': m_type}

#     print("Meilleure configuration de paramètres :", best_params)
#     print("Meilleur MSE obtenu :", round(best_mse, 4))

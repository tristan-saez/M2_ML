import numpy as np
from algorithms.NormalizeCarseats import PreProcessingCarSeats
from cvxopt import matrix, solvers

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

        P = matrix(np.outer(Y, Y) * K)
        q = matrix(-1 * np.ones(n))
        b = matrix(0.0)
        A = matrix(Y * 1.0, (1, n))

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


# Classifieur SVM
class svm_class:
    """One-vs-all classifier, utilise un ensemble de modèles svm_reg pour chaque classe dans la sortie."""
    def __init__(self, m_type='linear', C=6, gamma=1):
        self.models_dict = {}
        self.C = C
        self.gamma = gamma
        self.m_type = m_type

    def fit(self, X, Y):
        classes = set(Y)
        models_dict = dict.fromkeys(classes)
        for cl in classes:
            idxs = (Y == cl)
            Y_modified = -1 * np.ones(len(Y))
            Y_modified[idxs] = 1
            models_dict[cl] = svm_reg(m_type=self.m_type, C=self.C, gamma=self.gamma)
            models_dict[cl].fit(X, Y_modified.T)

        self.models_dict = models_dict

    def _predict(self, inpt):
        results_dict = dict.fromkeys(self.models_dict.keys())
        for cl in self.models_dict.keys():
            results_dict[cl] = self.models_dict[cl].predict(inpt)

        return max(results_dict, key=results_dict.get)

    def predict(self, inpt):
        inpt = np.array(inpt)
        if len(inpt.shape) == 1:
            return self._predict(inpt)
        else:
            res = np.zeros(len(inpt))
            for i in range(len(inpt)):
                res[i] = self._predict(inpt[i])
            return res


def accuracy_score(y_true, y_pred):
    """Calcule la précision du modèle."""
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true) if len(y_true) > 0 else 0

def svm(X_train, X_test, Y_train, Y_test):
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    model = svm_class(m_type='rbf', C=10000000, gamma=220)
    model.fit(X_train, Y_train)

    res_model = model.predict(X_test)
    return Y_test, res_model

# if __name__ == "__main__":
#     X_train, X_test, Y_train, Y_test = PreProcessingCarSeats('data/Carseats.csv', ',')
#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_test = X_test.values
#     Y_test = Y_test.values

#     param_grid = {
#         'C': [0.01, 0.1, 1, 10, 100, 150, 200, 400, 500, 600, 750, 850, 1000],
#         'gamma': [0.1, 1, 10, 20, 50, 100, 120, 160, 200, 220, 260, 500],
#         'm_type': ['rbf']  
#     }

#     best_accuracy = 0
#     best_params = {}

#     # Grid search sur les valeurs définies
#     for C in param_grid['C']:
#         for gamma in param_grid['gamma']:
#             for m_type in param_grid['m_type']:  # Boucle sur les types de modèles
#                 model = svm_class(m_type=m_type, C=C, gamma=gamma)
#                 model.fit(X_train, Y_train)
#                 accuracy = model.predict(X_test)
#                 accuracy_score_value = accuracy_score(Y_test, accuracy)
#                 print(f"C: {C}, gamma: {gamma}, m_type: {m_type}, accuracy: {round(accuracy_score_value * 100, 2)}%")
#                 if accuracy_score_value > best_accuracy: 
#                     best_accuracy = accuracy_score_value
#                     best_params = {'C': C, 'gamma': gamma, 'm_type': m_type}  
#     print("Meilleures valeurs de paramètres:", best_params)
#     print("Meilleure précision:", best_accuracy * 100, '%')
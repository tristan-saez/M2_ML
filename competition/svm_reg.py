import numpy as np
from cvxopt import matrix, solvers
from itertools import product

# Régressseur SVM
# Cette classe utilise la programmation quadratique pour calculer les vecteurs de poids et le biais associés, en fonction de l'entrée X (caractéristiques) et des étiquettes Y.
# Le modèle peut utiliser différents types de noyaux (kernel) pour ajuster la fonction d'hypothèse.
class svm_reg:
    """Régresseur de la machine à vecteurs de support, utilise la programmation quadratique pour résoudre le vecteur de poids et biais."""
    def __init__(self, m_type='linear', C=6, gamma=1):
        # Dans cette partie, nous initialisons les paramètres essentiels du modèle :
        # - 'C' est le paramètre de régularisation qui équilibre entre maximiser la marge de séparation
        # et minimiser l'erreur de classification.
        # - 'gamma' est un paramètre spécifique à certains noyaux comme le noyau gaussien (RBF).
        # - 'm_type' détermine le type de noyau utilisé (linear, polynomial, rbf, sigmoid).
        self.C = C
        self.gamma = gamma
        self.m_type = m_type
        self.alphas = None
        self.svs_y = None
        self.svs_x = None
        self.w = None
        self.nf = None
        self.b = None
        
        # En fonction du type de noyau choisi, nous définissons la fonction de noyau (kernel) appropriée.
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
    
    # Cette méthode implémente le noyau linéaire, qui calcule simplement le produit scalaire entre deux vecteurs d'entrée.
    def linear_kernal(self, x1, x2):
        return x1.T @ x2
    
    # Le noyau polynomial est défini comme (x1^T x2 + 1)^gamma, où gamma est un paramètre de la transformation.
    def polynomial_kernel(self, x1, x2):
        return (x1.T @ x2 + 1) ** self.gamma
    
    # Le noyau gaussien RBF applique une transformation exponentielle qui mesure la distance entre deux points
    # en utilisant la norme euclidienne, en fonction de 'gamma' et du nombre de caractéristiques.
    def rbf_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (self.nf * self.gamma ** 2))
    
    # Le noyau sigmoïde applique une transformation tanh au produit scalaire, en le modulant par 'gamma'.
    def sigmoid_kernel(self, x1, x2):
        return np.tanh(self.gamma * x1.T @ x2 + self.gamma)

    # La méthode 'fit' permet d'entraîner le modèle SVM en ajustant les paramètres internes comme 'w' et 'b' en utilisant
    # la programmation quadratique. Nous construisons la matrice de noyau 'K' pour capturer la similarité entre chaque paire de points.
    def fit(self, X: np.array, Y: np.array):
        """Initialise le modèle et calcule le vecteur de poids et b."""
        n, self.nf = X.shape
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])

        # Nous construisons les différentes matrices nécessaires pour le problème d'optimisation quadratique.
        P = matrix(K)
        q = matrix(-Y)
        A = matrix(np.ones((1, n)), tc='d')
        b = matrix(1.0)
        
        # Les matrices G et H représentent les contraintes d'inégalité
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
        
        # Calcul du vecteur de poids 'w' pour le cas linéaire, en faisant la somme pondérée des vecteurs d'entrée.
        self.w = np.zeros(self.nf)
        for alpha, y, x in zip(alphas, Y, X):
            self.w += alpha * y * x
        
        # Filtrage des support vectors, c'est-à-dire les vecteurs pour lesquels les valeurs d'alpha sont significatives.
        idxs = alphas > 1e-5
        self.svs_x = X[idxs]
        self.svs_y = Y[idxs]
        self.alphas = alphas[idxs]
        
        # Calcul de la valeur de biais 'b' en utilisant les vecteurs de support.
        self.b = np.mean([self.svs_y[i] - sum([alpha * y * self.kernel(x, self.svs_x[i])
                                               for alpha, y, x in zip(self.alphas, self.svs_y, self.svs_x)])
                          for i in range(len(self.svs_x))])
    
    # Méthode interne '_predict' qui prédit une valeur en utilisant le vecteur de poids 'w' et le biais 'b'.
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



"""
La classe « svm_reg » implémente un modèle de régression utilisant la méthode des Machines à Vecteurs de Support pour la régression. 
Le but principal de cette classe est de trouver une fonction qui, étant donné un ensemble de données d’entraînement, prédit avec précision une valeur continue 
pour des nouvelles observations. La classe prend en entrée les paramètres « C », « gamma », et « m_type », qui influencent respectivement la régularisation, 
la largeur du noyau (pour les noyaux non linéaires), et le type de noyau à utiliser pour transformer les données. 
L’apprentissage du modèle est réalisé en résolvant un problème d’optimisation quadratique.
Le paramètre « C » contrôle le compromis entre le degré d'ajustement du modèle aux données d’entraînement et la complexité du modèle. 
En d'autres termes, « C » influence la quantité de régularisation appliquée au modèle. Il agit comme une pénalité pour les erreurs d’entraînement, 
ce qui signifie qu’il détermine à quel point le modèle est autorisé à avoir des erreurs dans les prédictions lors de l'apprentissage.
Le paramètre « gamma » influence la complexité de la frontière de décision en contrôlant l’étendue d’influence d’un seul point de données. 
En d'autres termes, « gamma » détermine comment un point de données individuel affecte la décision du modèle. Il est particulièrement pertinent pour 
les noyaux non linéaires tels que le noyau RBF, polynomial, et sigmoïde.

"""
def svm(X_train, X_test, Y_train, Y_test):
    """
    Cette fonction implémente un modèle de régression SVM en utilisant un noyau gaussien radial (RBF).
    Elle prend en entrée les paramètres suivants :
    - 'X_train' : Un DataFrame ou tableau contenant les caractéristiques des données d'entraînement.
    - 'X_test' : Un DataFrame ou tableau contenant les caractéristiques des données de test.
    - 'Y_train' : Un DataFrame ou tableau contenant les étiquettes cibles associées aux données d'entraînement.
    - 'Y_test' : Un DataFrame ou tableau contenant les étiquettes cibles associées aux données de test.

    La fonction retourne :
    - 'Y_test' : Les étiquettes réelles des données de test.
    - 'predictions' : Les étiquettes prédites par le modèle SVM pour 'X_test'.
    """
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values
    
    
    model = svm_reg(m_type='linear', C=0.001, gamma=0.001)
    model.fit(X_train, Y_train)
    
    
    predictions = model.predict(X_test)
    return Y_test, predictions
    
# X_train, X_test, Y_train, Y_test = PreProcessing('Hitters_train.csv', 'Hitters_test.csv',',')
# y_test, predictions=svm(X_train, X_test, Y_train, Y_test)
# mse = mean_squared_error(Y_test, predictions)
# print(mse)


    
# if __name__ == "__main__":

#     X_train, X_test, Y_train, Y_test = PreProcessing('Hitters_train.csv', 'Hitters_test.csv',',')
#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_test = X_test.values
#     Y_test = Y_test.values

#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],  
#         'gamma': [0.001, 0.01, 0.1, 1, 10, 50, 100],     
#         'm_type': ['linear']  
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


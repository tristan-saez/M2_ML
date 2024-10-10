import numpy as np
from algorithms.NormalizeCarseats import PreProcessingCarSeats
from cvxopt import matrix, solvers

# Régressseur SVM
# Cette classe utilise la programmation quadratique pour calculer les vecteurs de poids et le biais associés, en fonction de l'entrée X (caractéristiques) et des étiquettes Y.
# Le modèle peut utiliser différents types de noyaux (kernel) pour ajuster la fonction d'hypothèse.
class svm_reg:
    """Régresseur de la machine à vecteurs de support, utilise la programmation quadratique pour résoudre le vecteur de poids et biais."""
    def __init__(self, m_type='linear', C=6, gamma=1):
        # Lors de l'initialisation de la classe, nous définissons différents hyperparamètres tels que le type de noyau, le paramètre de régularisation 'C' 
        # et le paramètre 'gamma'. Ces valeurs influencent la flexibilité et la complexité de notre modèle.
        self.C = C
        self.gamma = gamma
        self.m_type = m_type
        self.alphas = None
        self.svs_y = None
        self.svs_x = None
        self.w = None
        self.nf = None
        self.b = None
        
        # En fonction du type de noyau sélectionné, nous affectons la méthode de calcul correspondante.
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
        P = matrix(np.outer(Y, Y) * K)
        q = matrix(-1 * np.ones(n))
        b = matrix(0.0)
        A = matrix(Y * 1.0, (1, n))
        
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


# Classifieur SVM
# La classe « svm_class » implémente un classificateur SVM multiclasses basé sur plusieurs modèles « svm_reg ». 
# L'approche One-vs-All consiste à entraîner un régressseur SVM pour chaque classe afin de distinguer cette classe de toutes les autres. 
# Le modèle « svm_reg » est capable de prédire des valeurs réelles, ce qui est utile pour la classification lorsqu'il est utilisé dans une approche One-vs-All.
# Pour chaque classe unique dans « Y », on crée une instance de « svm_reg ». Par exemple, si « Y » a deux classes {0, 1}, deux modèles « svm_reg » seront créés.
# Les étiquettes « Y » sont modifiées pour chaque modèle : Les points appartenant à la classe actuelle sont étiquetés comme 1. 
# Les points appartenant aux autres classes sont étiquetés comme -1. La classe dont le modèle « svm_reg » retourne la plus grande valeur est choisie comme prédiction finale. 
# Si les valeurs de décision sont par exemple {-0.5, 0.8}, le modèle prédit la classe « 1 » car sa valeur (0.8) est la plus élevée.
class svm_class:
    """One-vs-all classifier, utilise un ensemble de modèles svm_reg pour chaque classe dans la sortie."""
    def __init__(self, m_type='linear', C=6, gamma=1):
        # Le dictionnaire "models_dict" sera utilisé pour stocker chaque modèle SVM distinct correspondant à chaque classe
        # dans le jeu de données. Les clés de ce dictionnaire seront les classes, et les valeurs seront les objets SVM correspondants.
        self.models_dict = {}
        # Les paramètres "C" et "gamma" sont essentiels pour contrôler le comportement du SVM. "C" détermine l'importance
        # de l'erreur de classification sur les données d'entraînement, tandis que "gamma" influe sur la portée de l'influence
        # des points d'entraînement individuels dans les noyaux non linéaires.
        self.C = C
        self.gamma = gamma
        self.m_type = m_type

    # La méthode "fit" est utilisée pour entraîner le classificateur. Nous recevons en entrée deux matrices :
    # "X" contient les caractéristiques des échantillons d'entraînement, et "Y" contient les étiquettes correspondantes.
    def fit(self, X, Y):
        classes = set(Y)
        models_dict = dict.fromkeys(classes)
        # Nous parcourons chaque classe afin de construire un modèle SVM spécifique pour chacune d'elles.
        # Pour chaque classe, nous marquons les échantillons appartenant à cette classe avec +1 et les autres échantillons avec -1.
        for cl in classes:
            idxs = (Y == cl)
            # Nous initialisons un vecteur "Y_modified" avec des valeurs -1 pour tous les échantillons.
            # Cela signifie que tous les échantillons ne faisant pas partie de la classe actuelle sont considérés comme négatifs.
            Y_modified = -1 * np.ones(len(Y))
            # Nous modifions ce vecteur pour attribuer +1 aux échantillons qui appartiennent à la classe actuelle "cl".
            Y_modified[idxs] = 1
            # Nous créons un nouvel objet "svm_reg" pour la classe actuelle. Chaque classe aura son propre modèle SVM.
            models_dict[cl] = svm_reg(m_type=self.m_type, C=self.C, gamma=self.gamma)
            # Nous entraînons le modèle sur les données d'entraînement "X" en utilisant le vecteur de sortie modifié "Y_modified".
            models_dict[cl].fit(X, Y_modified.T)

        self.models_dict = models_dict
    
    # La méthode "_predict" est une méthode interne qui effectue la prédiction pour un seul échantillon d'entrée.
    # Elle parcourt chaque modèle SVM et applique chaque modèle à l'échantillon en question pour calculer un score de décision.
    def _predict(self, inpt):
        results_dict = dict.fromkeys(self.models_dict.keys())
        for cl in self.models_dict.keys():
            results_dict[cl] = self.models_dict[cl].predict(inpt)
        # Nous retournons la classe qui a le score de décision le plus élevé.
        return max(results_dict, key=results_dict.get)
    
    # La méthode "predict" applique la méthode "_predict" pour effectuer une prédiction sur un vecteur ou une matrice de données.
    def predict(self, inpt):
        inpt = np.array(inpt)
        
        # Si l'entrée est un vecteur (c'est-à-dire un échantillon unique), nous utilisons "_predict" pour faire une seule prédiction.
        if len(inpt.shape) == 1:
            return self._predict(inpt)
        else:
            res = np.zeros(len(inpt))
            for i in range(len(inpt)):
                res[i] = self._predict(inpt[i])
            return res


#def accuracy_score(y_true, y_pred):
#    """Calcule la précision du modèle."""
#    correct_predictions = np.sum(y_true == y_pred)
#    return correct_predictions / len(y_true) if len(y_true) > 0 else 0

def svm(X_train, X_test, Y_train, Y_test):
    """
    Fonction pour entraîner un modèle SVM multi-classe de type one-vs-all et faire des prédictions sur des données de test.

    Elle prend en entrée les paramètres suivants :
    - 'X_train' : Un DataFrame ou tableau contenant les caractéristiques des données d'entraînement.
    - 'X_test' : Un DataFrame ou tableau contenant les caractéristiques des données de test.
    - 'Y_train' : Un DataFrame ou tableau contenant les étiquettes cibles associées aux données d'entraînement.
    - 'Y_test' : Un DataFrame ou tableau contenant les étiquettes cibles associées aux données de test.

    Sorties :
    - Y_test : Les étiquettes réelles des données de test.
    - res_model : Les classes prédites pour les échantillons de X_test par le modèle SVM entraîné.

    """
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
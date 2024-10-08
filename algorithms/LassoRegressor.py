import pandas as pd
import numpy as np

# Classe Lasso Regression
class LassoRegression():
    """
    Une classe pour la régression L1 (ou LASSO)
        Attributs :
                    iterations (int) : nombre d'itérations d'exécution de l'algorithme de gradient
                    learning_rate (float) : taille des pas à chaque mise à jour des paramètres
                    l1_penalty (float) : coefficient de régularisation (lambda)
                    m (int) : nombre total des données d'entraînement
                    n (int) : nombre de features du dataset
                    w (numpy array) : vecteur de taille n des poids associés à chaque feature du dataset
                    b (float) : biais du modèle
                    X (numpy array) : données d'entrée d'entraînement
                    Y (numpy array) : valeurs cibles des données d'entraînement
        Méthodes :
                    fit : entraîne le modèle
                    update_weights : met à jour le poids de l'algorithme de descente
                    predict :
    """
    #Constructeur
    def __init__(self, iterations, learning_rate, l1_penalty):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.m, self.n = None, None
        self.w = None
        self.b = 0
        self.X = None
        self.Y = None

    # Méthode d'entraînement
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X.to_numpy()
        self.Y = Y.to_numpy()
        for i in range(self.iterations):
            self.update_weights()
        return self
        
    # Met à jour les poids pour l'algorithme de descente (gradient descent)
    # Correspond à l'application de la pénalité L1
    # Force à mettre certains poids à zéro
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # Calcul de la dérivée partielle (ou gradient) de l'erreur par rapport à chaque poids w[j]
        dw = np.zeros(self.n)
        for j in range(self.n):
            # Détermine le signe de lambda
            if self.w[j] > 0:
                # Correspond à la fonction de coût classique + ou - lambda
                dw[j] = (2 * (self.X[:,j]).dot(self.Y - Y_pred) + self.l1_penalty) / self.m
            else:
                dw[j] = (2 * (self.X[:,j]).dot(self.Y - Y_pred) - self.l1_penalty) / self.m
        # Calcul de la dérivée partielle (gradient) de l'erreur par rapport au biais b
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        # Mise à jour des poids
        # Permet de faire descendre l'erreur
        # Ajuste progressivement le poids et le biais à chaque itération
        self.w = self.w - self.learning_rate * dw
        # Mise à jour du biais
        self.b = self.b - self.learning_rate * db
        return self
    
    # Fonction d'hypothèse h(x)
    def predict(self, X):
        return X.dot(self.w) + self.b


def lasso_regressor(X_train, X_test, Y_train, Y_test):
    """
    Crée un objet LassoRegression, entraîne le modèle, l'évalue et le compare avec la méthode associée scikit-learn.
        Paramètres :
                    X_train : Features du set d'entraînement
                    Y_train : Feature à prédire du set d'entraînement
                    X_test : Features du set de test
                    Y_test : Feature réelle du set d'entraînement
        Retourne :
                    Y_test
                    Y_pred : Valeur cible prédite du set de test par le modèle
    """
    # Entraînement du modèle
    model = LassoRegression(
        iterations=1000,
        learning_rate=0.01,
        l1_penalty=500
    )
    model.fit(X_train, Y_train)

    # Prédictions sur le set de test
    Y_pred = model.predict(X_test)

    # Vérifications sur les 3 dernières valeurs du set de test
    print(f"Predicted values : {np.round(Y_pred[:3],2)}")
    print(f"Real balues : {Y_test[:3]}")

    return Y_test, Y_pred
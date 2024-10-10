"""
Ce fichier utilise l'algorithme arbre de décision (utilisant la classification) 
author : celine dussuelle 
creation 01/10/2024"""

# To do's :
# Traduire les commentaires
# completer le rapport avec mes données en utilisant le main 
# foret aleatoire pas vraiment aléatoire. => problème ? 

#################################################################################################
####################################### IMPORTATIONS ############################################
#################################################################################################

#import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px
#import pprint 
#from NormalizeCarseats import PreProcessingCarSeats
from sklearn.tree import DecisionTreeClassifier 

#################################################################################################
######################################### CLASSES ###############################################
#################################################################################################



class Node():
    """
    Classe représentant un noeud dans l'arbre de décision
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initialise une nouvelle instance de la classe Node.

        Arguments :
            feature : La caractéristique utilisée pour diviser à ce nœud. Par défaut à None.
            threshold : Le seuil utilisé pour diviser à ce nœud. Par défaut à None.
            left : Le nœud enfant gauche. Par défaut à None.
            right : Le nœud enfant droit. Par défaut à None.
            gain : Le gain de la division. Par défaut à None.
            value : Si ce nœud est une feuille, cet attribut représente la valeur prédite 
                    pour la variable cible. Par défaut à None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DecisionTree():
    """
    Un classificateur d'arbre de décision pour des problèmes de classification binaire.
    """

    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructeur de la classe DecisionTree.

        Paramètres :
        min_samples (int) : Nombre minimum d'échantillons requis pour diviser un nœud interne.
        max_depth (int) : Profondeur maximale de l'arbre de décision.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
        """
        Divise le jeu de données donné en deux sous-ensembles selon la caractéristique et le seuil spécifiés.

        Paramètres :
            dataset (ndarray) : Jeu de données d'entrée.
            feature (int) : Indice de la caractéristique sur laquelle diviser.
            threshold (float) : Valeur seuil pour diviser la caractéristique.

        Renvoie :
            left_dataset (ndarray) : Sous-ensemble du jeu de données avec des valeurs inférieures ou égales au seuil.
            right_dataset (ndarray) : Sous-ensemble du jeu de données avec des valeurs supérieures au seuil.
        """
        
        # Creation de numpy arrays pour stocker les données de gauche et les données de droite. 
        left_dataset = []
        right_dataset = []
        
        # Boucle sur chaque ligne des données et les séparent en fonction de 'feature' (la caractéristique)
        # et threshold (le seuil)
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        
        # Conversion des données de gauche et des données de droite en numpy arrays. 
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        Calcule l'entropie des valeurs d'étiquettes fournies.

        Paramètres :
            y (ndarray) : Valeurs d'étiquettes en entrée.

        Renvoie :
            entropy (float) : Entropie des valeurs d'étiquettes fournies.
        """
        entropy = 0

        
        # Trouver les valeurs de label uniques dans y et boucle sur chaque valeur
        labels = np.unique(y)
        for label in labels:
            
            # Trouver les exemples dans y qui possèdent le label actuel
            label_examples = y[y == label]           
            # Calcul du ratio du label actuel dans y
            pl = len(label_examples) / len(y)
            # Calcul de l'entropie en utilisant l'actuel label et ratio 
            entropy += -pl * np.log2(pl)

        # Retour de la valeur entropie
        return entropy

    def information_gain(self, parent, left, right):
        """
        Calcule le gain d'information en divisant le jeu de données parent en deux sous-ensembles.

        Paramètres :
            parent (ndarray) : Jeu de données parent en entrée.
            left (ndarray) : Sous-ensemble du jeu de données parent après division selon une caractéristique.
            right (ndarray) : Sous-ensemble du jeu de données parent après division selon une caractéristique.

        Renvoie :
            information_gain (float) : Gain d'information de la division.
        """
        # Initialisation de la variable information_gain à 0
        information_gain = 0
        # Calcul de l'entropie du jeu de données parent
        parent_entropy = self.entropy(parent)
        # Calcul du poids de chaque noeud (droite et gauche)
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        # calcul de l'entropie pour chaque noeud
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # calcul de l'entropie pondérée
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        # Calcul du gain d'information de la division 
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset, num_samples, num_features):
        """
        Trouve la meilleure division pour le jeu de données donné.

        Arguments :
            dataset (ndarray) : Le jeu de données à diviser.
            num_samples (int) : Le nombre d'échantillons dans le jeu de données.
            num_features (int) : Le nombre de caractéristiques dans le jeu de données.

        Renvoie :
            dict : Un dictionnaire avec l'indice de la meilleure caractéristique pour diviser, le seuil, 
                le gain, ainsi que les sous-ensembles gauche et droit.
        """
        
        # initialisation d'un dictionnaire pour stocker les meilleures valeurs pour séparer les données
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        # Boucle sur toutes les caractéristiques
        for feature_index in range(num_features):
            
            # récupération de la caractéristique 
            feature_values = dataset[:, feature_index]
            # Récupération des valeurs uniques pour cette caractéristique
            thresholds = np.unique(feature_values)
            # boucle sur toutes les valeurs de la caractéristique 
            for threshold in thresholds:
                # Séparation des données en 2
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                # Vérifier si l'un des dataset est vide
                if len(left_dataset) and len(right_dataset):
                    # Récupération des valeur y pour le noeud parent, le noeud de de gauche et le noeud de droite                    
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    
                    # Calcul du gain d'information basé sur les valeurs y
                    information_gain = self.information_gain(y, left_y, right_y)
                    
                    # actualise la meilleure separation de données si les conditions sont bonnes
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split

    
    def calculate_leaf_value(self, y):
        """
       Calcule la valeur la plus fréquente dans la liste des valeurs y données.

        Arguments :
            y (list) : La liste des valeurs y.

        Renvoie :
            La valeur la plus fréquente dans la liste.
        """
        y = list(y)
        # Récupération de la valeur la plus fréquente dans la liste des valeur y
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth=0):
        """
        Construit récursivement un arbre de décision à partir du jeu de données donné.

        Arguments :
            dataset (ndarray) : Le jeu de données pour construire l'arbre.
            current_depth (int) : La profondeur actuelle de l'arbre.

        Renvoie :
            Node : Un noeud de décision si l'arbre est toujours en construction. Un noeud feuille si ce n'est pas le cas
        """
        
        # Séparation du dataset en 2 (X et y)
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # Si le nombre d'échantillons est supérieur ou égal à self.min_samples et que la profondeur actuelle est 
        # inférieure ou égale à self.max_depth, l'algorithme continue de diviser. Sinon, il arrête de diviser.
        
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Récupération de la meilleure séparation
            best_split = self.best_split(dataset, n_samples, n_features)
            # Vérifier si le gain n'est pas 0
            if best_split["gain"]>0:
                # continuer à diviser le dataset gauche et le dataset droit. Augmention de la profondeur de l'arbre
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # retourne la valeur du noeud
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # calcul de la valeur du noeud feuille
        leaf_value = self.calculate_leaf_value(y)
        # retourne la valeur du noeud feuille
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        """
        Construit et ajuste l'arbre de décision aux valeurs X et y fournies.

        Arguments :
            X (ndarray) : La matrice des caractéristiques.
            y (ndarray) : Les valeurs cibles.
        """
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Prédit les étiquettes de classe pour chaque instance dans la matrice des caractéristiques X.

        Arguments :
            X (ndarray) : La matrice des caractéristiques pour laquelle faire des prédictions.

        Renvoie :
            list : Une liste des étiquettes de classe prédites. 
        """
        
        # Initialisation d'une liste pour stocker les prédictions
        predictions = []
        # Pour chaque instance de X, faire une prédiction en traversant l'arbre. 
        for x in X:     
            prediction = self.make_prediction(x, self.root)
            # Ajout de la prédiction dans la liste des prédictions
            predictions.append(prediction)

        # Convertion de la liste en numpy array et la retourne
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        """
        Parcourt l'arbre de décision pour prédire la valeur cible pour le vecteur de caractéristiques donné.

        Arguments :
            x (ndarray) : Le vecteur de caractéristiques pour lequel prédire la valeur cible.
            node (Node) : Le nœud actuel évalué.

        Renvoie :
            La valeur cible prédite pour le vecteur de caractéristiques donné.
        """
       
        # Si le noeud a une valeur c'est une feuille donc on extrait sa valeur. 
        
        if node.value != None: 
            return node.value
        else:
            # Sinon c'est un noeud donc on récupère la caractéristique et on parcourt l'arbre en fonction de celle-ci
            feature = x[node.feature]
                        
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
            
#################################################################################################
######################################### FONCTIONS #############################################
#################################################################################################



def accuracy(y_true, y_pred):
    """
    Calcule la précision d'un modèle de classification.

    Paramètres :
    ----------
        y_true (numpy array) : Un tableau numpy des étiquettes réelles pour chaque point de données.
        y_pred (numpy array) : Un tableau numpy des étiquettes prédites pour chaque point de données.

    Renvoie :
    ----------
        float : La précision du modèle.
    """
   
    
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 


def balanced_accuracy(y_true, y_pred):
    """Calcule la précision équilibrée pour un problème de classification multi-classe.

    Paramètres
    ----------
        y_true (numpy array) : Un tableau numpy des étiquettes réelles pour chaque point de données.
        y_pred (numpy array) : Un tableau numpy des étiquettes prédites pour chaque point de données.

    Renvoie
    -------
        balanced_acc : La précision équilibrée du modèle.
        
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Récupération du nombre de classes
    n_classes = len(np.unique(y_true))

    # Initialise une liste pour stocker la sensibilité et la spécificité de chaque classe 
    
    sen = []
    spec = []
    # Boucle sur chaque classe
    for i in range(n_classes):
        # Créer un masque pour les valeurs vraies et prédites pour la classe i
        mask_true = y_true == i
        mask_pred = y_pred == i

        # Calcul des vrai positif, vrai négatif, faux positif et faux négatif
        
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # Calculer la sensibilité (taux vrai positif) et la spécificité (taux vrai négatif)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # Ajout des deux valeurs (sensibilité et spécificité) dans une liste
        sen.append(sensitivity)
        spec.append(specificity)
    # Calculer la précision équilibrée comme moyenne de la sensibilité et de la spécificité pour chaque classe
    average_sen =  np.mean(sen)
    average_spec =  np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc


def decision_tree(X_train, X_test, Y_train, Y_test) :
    """Crée un objet LassoRegression, entraîne le modèle à partir de la base d'entraînement et prédit la valeur de sortie à partir de la base de test.
        Paramètres :
                    X_train : Features du set d'entraînement
                    Y_train : Feature à prédire du set d'entraînement
                    X_test : Features du set de test
                    Y_test : Feature réelle du set d'entraînement
        Retourne :
                    Y_test : Valeur cible réelle
                    Y_pred : Valeur cible prédite du set de test par le modèle"""
    # Création du model
    model = DecisionTree(2, 2)

    # Application du model sur les données d'entrainement 
    model.fit(X_train, Y_train.to_frame())
    
    # Calcul des prédictions avec les données de test
    predictions = model.predict(X_test.to_numpy())

    # calcul de la précision
    print(f"Model's Accuracy: {accuracy(Y_test, predictions)}")
    print(f"Model's Balanced Accuracy: {balanced_accuracy(Y_test.to_numpy(), predictions)}")
    return Y_test, predictions


#################################################################################################
#################################### DECISION TREE SCIKIT-LEARN #################################
#################################################################################################


def decision_tree_sklearn(X_train, X_test, Y_train, Y_test):
    """
    Crée un objet LassoRegression, entraîne le modèle à partir de la base d'entraînement et prédit la 
    valeur de sortie à partir de la base de test.
        Paramètres :
                    X_train : Features du set d'entraînement
                    Y_train : Feature à prédire du set d'entraînement
                    X_test : Features du set de test
                    Y_test : Feature réelle du set d'entraînement
        Retourne :
                    Y_test : Valeur cible réelle
                    Y_pred : Valeur cible prédite du set de test par le modèle
    """
     # entrainement du model
    clf = DecisionTreeClassifier(random_state=1)  
    clf.fit(X_train, Y_train)

    # Prédiction
    Y_pred = clf.predict(X_test)

    return Y_test, Y_pred
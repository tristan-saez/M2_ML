# Code crée à l'aide des informations fournis sur le site suivant :
#   https://www.kaggle.com/code/fareselmenshawii/random-forest-from-scratch#Decision-Tree-Implementation


from algorithms.DecisionTree import DecisionTree
from algorithms.NormalizeCarseats import PreProcessingCarSeats

# from DecisionTree import DecisionTree
# from NormalizeCarseats import PreProcessingCarSeats
import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier


class random_forest_classifier:

    def __init__(self, nb_arbres=100, profondeur_max=20, echantillons_max=5):
        self.nb_arbres = nb_arbres
        self.profondeur_max = profondeur_max
        self.echantillons_max = echantillons_max
        self.liste_arbres = []

    def bootstrap_echantillons(self, data):
        # Récupère la longueur du set de donnée
        n_echantillons = data.shape[0]

        # Génère des indices aléatoires pour le nouveau set de donnée
        np.random.seed(8)
        rng = np.random.default_rng()
        indices = rng.choice(
            a=n_echantillons, size=n_echantillons, replace=True)

        # Return the bootstrapped dataset sample using the generated indices.
        data_echantillon = data.iloc[indices]

        return data_echantillon

    def fit(self, X, Y):
        data = pd.concat([X, Y],
                         axis=1)

        for _ in range(self.nb_arbres):
            # Nom de la classe appelé en PlaceHolder tant que le DecisionTree n'est pas finalisé
            arbre_solo = DecisionTree(
                self.echantillons_max, self.profondeur_max)

            # Echantillonne en remplacant depuis notre set de donnée entier (X et Y combinés)
            echantillon_data = self.bootstrap_echantillons(data)

            # Redécoupe notre set de donnée
            X_echantillon, Y_echantillon = echantillon_data.iloc[:,
                                                                 :-1], echantillon_data.iloc[:, -1:]

            # print(X_echantillon, Y_echantillon)

            arbre_solo.fit(X_echantillon, Y_echantillon)

            self.liste_arbres.append(arbre_solo)

        return self

    def prediction_la_plus_commune(self, liste_predictions):
        my_list = list(liste_predictions)

        return max(set(my_list), key=my_list.count)

    def predict(self, X):
        # Récupère une liste de prédiction par arbre du tableau de donnée X
        # EX : [prediction_X_table_1, prediction_X_table_2, ... , prediction_X_table_{max_arbres}]
        liste_predictions = np.array([arbre_solo.predict(X)
                                      for arbre_solo in self.liste_arbres])

        predictions = np.swapaxes(liste_predictions, 0, 1)

        # Récupère la liste des labels apparaissant le plus pour chaque donnée
        labels = np.array([self.prediction_la_plus_commune(prediction_solo_X)
                           for prediction_solo_X in predictions])

        return labels


def accuracy(y_true, y_pred):
    """
    Computes the accuracy of a classification model.

    Parameters:
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns:
    ----------
        float: The accuracy of the model
    """

    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples)


def balanced_accuracy(y_true, y_pred):
    """Calculate the balanced accuracy for a multi-class classification problem.

    Parameters
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns
    -------
        balanced_acc : The balanced accuracyof the model

    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Get the number of classes
    n_classes = len(np.unique(y_true))

    # Initialize an array to store the sensitivity and specificity for each class
    sen = []
    spec = []
    # Loop over each class
    for i in range(n_classes):
        # Create a mask for the true and predicted values for class i
        mask_true = y_true == i
        mask_pred = y_pred == i

        # Calculate the true positive, true negative, false positive, and false negative values
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # Calculate the sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # Store the sensitivity and specificity for class i
        sen.append(sensitivity)
        spec.append(specificity)
    # Calculate the balanced accuracy as the average of the sensitivity and specificity for each class
    average_sen = np.mean(sen)
    average_spec = np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc


def random_forest(X_train, X_test, Y_train, Y_test):
    """
    Crée un objet Random_Forest_Classifier, entraîne le modèle, l'évalue et le compare avec la méthode associée scikit-learn.
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
    model = random_forest_classifier(
        nb_arbres=100,
        profondeur_max=15,
        echantillons_max=2
    )
    model.fit(X_train, Y_train.to_frame())

    # Prédictions sur le set de test
    Y_pred = model.predict(X_test.to_numpy())

    # Vérifications sur les 3 dernières valeurs du set de test
    # print(f"Model's Accuracy: {accuracy(Y_test, Y_pred)}")
    # print(
    #     f"Model's Balanced Accuracy: {balanced_accuracy(Y_test.to_numpy(), Y_pred)}")

    return Y_test, Y_pred


def random_forest_sklearn(X_train, X_test, Y_train, Y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    # Vérifications sur les 3 dernières valeurs du set de test
    # print(f"Model's Accuracy: {accuracy(Y_test, Y_pred)}")
    # print(
    #     f"Model's Balanced Accuracy: {balanced_accuracy(Y_test.to_numpy(), Y_pred)}")

    return Y_test, Y_pred


def test_random_forest_scratch():
    X_train, X_test, Y_train, Y_test = PreProcessingCarSeats(
        'data/Carseats.csv', ',')

    random_forest_scratch(X_train, X_test, Y_train, Y_test)


def test_random_forest_sklearn():
    X_train, X_test, Y_train, Y_test = PreProcessingCarSeats(
        'data/Carseats.csv', ',')

    random_forest_sklearn(X_train, X_test, Y_train, Y_test)


if __name__ == '__main__':
    test_random_forest_scratch()
    # test_random_forest_sklearn()

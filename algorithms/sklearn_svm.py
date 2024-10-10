from sklearn.svm import SVC
from sklearn.svm import SVR

# Nous utilisons ici la SVM pour effectuer une tâche de classification en utilisant sklearn.
def svm_classification(X_train, X_test, Y_train, Y_test):

    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values
    

    model = SVC(kernel='rbf', C=1000, gamma=0.01)
    model.fit(X_train, Y_train)

    res_model = model.predict(X_test)
    return Y_test,res_model

#  Nous utilisons ici la SVM pour effectuer une tâche de régression en utilisant sklearn.
def svm_reg(X_train, X_test, Y_train, Y_test):

    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values
    

    model = SVR(kernel='rbf', C=1, gamma=1)
    model.fit(X_train, Y_train)

    res_model = model.predict(X_test)
    return Y_test, res_model
    
############################################SVC######################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from algorithms.NormalizeCarseats import PreProcessingCarSeats
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# if __name__ == "__main__":
#     # Chargement des données
#     X_train, X_test, Y_train, Y_test = PreProcessingCarSeats('data/Carseats.csv', ',')
#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_test = X_test.values
#     Y_test = Y_test.values

#     # Définition des paramètres à tester
#     C_values = [0.1, 1, 10, 100, 1000, 10000]
#     gamma_values = [0.01, 0.1, 1, 10, 100, 200]
#     kernel_types = ['rbf']

#     best_accuracy = 0
#     best_params = {}

#     # Boucle sur toutes les combinaisons de C, gamma et kernel
#     for C in C_values:
#         for gamma in gamma_values:
#             for kernel in kernel_types:
#                 # Création et entraînement du modèle SVM
#                 model = SVC(C=C, gamma=gamma, kernel=kernel)
#                 model.fit(X_train, Y_train)

#                 # Prédiction et évaluation du modèle
#                 res_model = model.predict(X_test)
#                 accuracy = round(accuracy_score(Y_test, res_model) * 100, 2)

#                 print(f"C: {C}, gamma: {gamma}, kernel: {kernel}, accuracy: {accuracy}%")

#                 # Vérification si cette configuration est la meilleure
#                 if accuracy > best_accuracy:
#                     best_accuracy = accuracy
#                     best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}

#     print('Meilleurs paramètres :', best_params)
#     print('Meilleure précision sur l\'ensemble de test :', best_accuracy, '%')

########################################SVR###################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from algorithms.NormalizeCarseats import PreProcessingCarSeats
# from sklearn.svm import SVR
# from algorithms.NormalizeOzone import PreProcessingOzone
# from sklearn import metrics

# if __name__ == "__main__":
#     # Chargement des données
#     X_train, X_test, Y_train, Y_test = PreProcessingOzone('data/ozone_complet.txt', ';')
#     X_train = X_train.values
#     Y_train = Y_train.values
#     X_test = X_test.values
#     Y_test = Y_test.values

#     # Définition des paramètres à tester
#     C_values = [0.1, 1, 10, 100, 1000, 10000]
#     gamma_values = [0.01, 0.1, 1, 10, 100, 200]
#     kernel_types = ['rbf']

#     best_accuracy = 0
#     best_params = {}

#     # Boucle sur toutes les combinaisons de C, gamma et kernel
#     for C in C_values:
#         for gamma in gamma_values:
#             for kernel in kernel_types:
#                 # Création et entraînement du modèle SVM
#                 model = SVR(C=C, gamma=gamma, kernel=kernel)
#                 model.fit(X_train, Y_train)

#                 # Prédiction et évaluation du modèle
#                 res_model = model.predict(X_test)
#                 accuracy = round(accuracy_score(Y_test, res_model) * 100, 2)

#                 print(f"C: {C}, gamma: {gamma}, kernel: {kernel}, accuracy: {accuracy}%")

#                 # Vérification si cette configuration est la meilleure
#                 if accuracy > best_accuracy:
#                     best_accuracy = accuracy
#                     best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}

#     print('Meilleurs paramètres :', best_params)
#     print('Meilleure précision sur l\'ensemble de test :', best_accuracy, '%')
###############################################################################################



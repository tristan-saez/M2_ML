import pandas as pd
from timeit import default_timer as timer
from algorithms import DecisionTree
from algorithms import RandomForest
from algorithms import RidgeRegressor
from algorithms import LassoRegressor
from competition import svm_classification
from competition import svm_reg
from algorithms import CheckScore
from algorithms import sklearn_svm
from competition import Normalize_reg
from competition import Normalize_classification


def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    

    # Effectue les étapes de pre-processing
    x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats = (
        Normalize_classification.PreProcessing("competition/data/hitters_train.csv", "competition/data/hitters_test.csv",","))

    # Effectue les étapes de pre-processing
    x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone = (
        Normalize_reg.PreProcessing("competition/data/hitters_train.csv", "competition/data/hitters_test.csv",","))

    # Permet à l'utilisateur de choisir l'algorithme à entraîner
    print("=" * 50, "\nCHOIX ALGORITHME\n"+"=" * 50+"\n1. Arbre de décisions\n2. Forêts aléatoires\n"
                                                    "3. Régression ridge\n4. Régression lasso\n"
                                                    "5. SVM (Classification)\n6. SVM (Regression)")

    choice = int(input("\n>"))

    if choice == 1:
        print("=" * 50, "\nARBRE DE DÉCISION\n" + "=" * 50)
        if model_choice() == 1:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = DecisionTree.decision_tree(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        else:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = DecisionTree.decision_tree_sklearn(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)

        # Arrête le chronomètre de temps d'apprentissage
        end_time = timer()
        # Vérifie les différents scores en fonction du modèle de Machine Learning (regression/classification)
        CheckScore.check_score("classifier", y_test, y_pred, start_time, end_time)
        
    elif choice == 2:
        print("=" * 50, "\nFORÊTS ALÉATOIRES\n" + "=" * 50)
        if model_choice() == 1:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = RandomForest.random_forest(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        else:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = RandomForest.random_forest(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)

        # Arrête le chronomètre de temps d'apprentissage
        end_time = timer()
        # Vérifie les différents scores en fonction du modèle de Machine Learning (regression/classification)
        CheckScore.check_score("classifier", y_test, y_pred, start_time, end_time)
        
    elif choice == 3:
        # Lance le chronomètre de temps d'apprentissage
        start_time1 = timer()
        #On appelle la classe RidgeRegressor pour créer le modèle. Ce dernier est paramétré pour apprendre sur 1000 epochs avec 
        #un taux d'apprentissage de 0.01. De plus, le coefficient de pénalité associer à l'algorithme de Ridge est de 5
        model = RidgeRegressor.RidgeRegressor( iterations = 1000,learning_rate = 0.01, l2_penality = 5 ) 
        #On entraine le modèle avec les précédents paramètres et la base d'entrainement
        model.fit(x_train_ozone, y_train_ozone) 
        #On prédit les valeurs de notre base de test à partir du modèle entraîné
        y_pred = model.predict(x_test_ozone)
        y_test=y_test_ozone
        end_time1 = timer()
        CheckScore.check_score("regressor", y_test, y_pred, start_time1, end_time1)

        # Lance le chronomètre de temps d'apprentissage
        start_time2 = timer()
        #On appel la classe RidgeRegressorSklearn, qui vient créer le modèle. Ce modèle utilise les méthodes issues de sklearn
        #De plus, le coefficient de pénalité associer à l'algorithme de Ridge est toujours de 5 pour comparer au modèle précédent.
        model2 = RidgeRegressor.RidgeRegressorSklearn(l2_penality = 5) 
        #On entraine le modèle avec les paramètres précédents et la base d'entrainement
        model2.fit(x_train_ozone, y_train_ozone) 
      
        #On prédit les valeurs de notre base de test à partir du modèle entraîné
        y_pred2 = model2.predict( x_test_ozone )     
        y_test2=y_test_ozone
        end_time2 = timer()
        CheckScore.check_score("regressor", y_test2, y_pred2, start_time2, end_time2)
        
    elif choice == 4:
        print("=" * 50, "\nRÉGRESSION LASSO\n" + "=" * 50)
        if model_choice() == 1:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = LassoRegressor.lasso_regressor(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        else:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = LassoRegressor.lasso_regressor_sklearn(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)

        # Arrête le chronomètre de temps d'apprentissage
        end_time = timer()
        # Vérifie les différents scores en fonction du modèle de Machine Learning (regression/classification)
        CheckScore.check_score("regressor", y_test, y_pred, start_time, end_time)

    elif choice == 5:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        if model_choice() == 1:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = svm_classification.svm(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        else:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = sklearn_svm.svm_classification(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)

        # Arrête le chronomètre de temps d'apprentissage
        end_time = timer()
        # Vérifie les différents scores en fonction du modèle de Machine Learning (regression/classification)
        CheckScore.check_score("classifier", y_test, y_pred, start_time, end_time)

    elif choice == 6:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        if model_choice() == 1:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = svm_reg.svm(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        else:
            # Lance le chronomètre de temps d'apprentissage
            start_time = timer()
            y_test, y_pred = sklearn_svm.svm_reg(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)

        # Arrête le chronomètre de temps d'apprentissage
        end_time = timer()
        # Vérifie les différents scores en fonction du modèle de Machine Learning (regression/classification)
        CheckScore.check_score("regressor", y_test, y_pred, start_time, end_time)

    else:
        print("Choix non reconnu")


def model_choice():
    """

    Demande à l'utilisateur de choisir entre le modèle "from scratch" et le modèle scikit-learn

    :return: modèle choisi (en entier : 1 ou 2)
    """
    print("\nSelectionnez le choix d'algortihme\n1. From \"scratch\"\n2. Scikit-learn\n")
    return int(input(">"))


if __name__ == '__main__':
    main()

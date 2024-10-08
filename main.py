import pandas as pd
from algorithms import DecisionTree
from algorithms import RandomForest
from algorithms import RidgeRegressor
from algorithms import LassoRegressor
from algorithms import SVM_Carseats
from algorithms import SVM_Ozone
from algorithms import NormalizeOzone
from algorithms import NormalizeCarseats
from algorithms import CheckScore
from algorithms import sklearn_svm


def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats = (
        NormalizeCarseats.PreProcessingCarSeats("data/Carseats.csv", ","))

    x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone = (
        NormalizeOzone.PreProcessingOzone("data/ozone_complet.txt", ";"))

    print("=" * 50, "\nCHOIX ALGORITHME\n"+"=" * 50+"\n1. Arbre de décisions\n2. Forêts aléatoires\n"
                                                    "3. Régression ridge\n4. Régression lasso\n"
                                                    "5. SVM (Classification)\n6. SVM (Regression)")

    choice = int(input("\n>"))

    if choice == 1:
        print("=" * 50, "\nARBRE DE DÉCISION\n" + "=" * 50)
        y_test, y_pred = DecisionTree.decision_tree(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        CheckScore.check_score("classifier", y_test, y_pred)
    elif choice == 2:
        print("=" * 50, "\nFORÊTS ALÉATOIRES\n" + "=" * 50)
        y_test, y_pred = RandomForest.random_forest(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        CheckScore.check_score("classifier", y_test, y_pred)
    elif choice == 3:
        print("=" * 50, "\nRÉGRESSION RIDGE\n" + "=" * 50)
        y_test, y_pred = RidgeRegressor.ridge_regressor(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        CheckScore.check_score("regressor", y_test, y_pred)
    elif choice == 4:
        print("=" * 50, "\nRÉGRESSION LASSO\n" + "=" * 50)
        if model_choice() == 1:
            y_test, y_pred = LassoRegressor.lasso_regressor(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        else:
            y_test, y_pred = LassoRegressor.lasso_regressor_sklearn(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        CheckScore.check_score("regressor", y_test, y_pred)
    elif choice == 5:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        if model_choice() == 1:
            y_test, y_pred = SVM_Carseats.svm(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        else:
            y_test, y_pred = sklearn_svm.svm_classification(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
        CheckScore.check_score("classifier", y_test, y_pred)
    elif choice == 6:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        if model_choice() == 1:
            y_test, y_pred = SVM_Ozone.svm(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        else:
            y_test, y_pred = sklearn_svm.svm_reg(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
        CheckScore.check_score("regressor", y_test, y_pred)
    else:
        print("Choix non reconnu")


def model_choice():
    print("\nSelectionnez le choix d'algortihme\n1. From \"scratch\"\n2. Scikit-learn\n")
    return int(input(">"))


if __name__ == '__main__':
    main()

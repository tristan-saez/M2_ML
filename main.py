import pandas as pd
from algorithms import DecisionTree
from algorithms import RandomForest
from algorithms import RidgeRegressor
from algorithms import LassoRegressor
from algorithms import SVM_karim
from algorithms import SVM_tristan
from algorithms import NormalizeOzone
from algorithms import NormalizeCarseats


def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats = (
        NormalizeCarseats.PreProcessingCarSeats("data/Carseats.csv"))

    x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone = (
        NormalizeOzone.PreProcessingOzone("data/ozone_complet.txt"))

    print("=" * 50, "\nCHOIX ALGORITHME\n"+"=" * 50+"\n1. Arbre de décisions\n2. Forêts aléatoires\n"
                                                    "3. Régression ridge\n4. Régression lasso\n"
                                                    "5. SVM (karim)\n6. SVM (tristan)")
    choice = int(input("\n>"))


    if choice == 1:
        print("=" * 50, "\nARBRE DE DÉCISION\n" + "=" * 50)
        DecisionTree.decision_tree(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
    elif choice == 2:
        print("=" * 50, "\nFORÊTS ALÉATOIRES\n" + "=" * 50)
        RandomForest.random_forest(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
    elif choice == 3:
        print("=" * 50, "\nRÉGRESSION RIDGE\n" + "=" * 50)
        RidgeRegressor.ridge_regressor(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
    elif choice == 4:
        print("=" * 50, "\nRÉGRESSION LASSO\n" + "=" * 50)
        LassoRegressor.lasso_regressor(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
    elif choice == 5:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        SVM_karim.svm(x_train_carseats, x_test_carseats, y_train_carseats, y_test_carseats)
    elif choice == 6:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        SVM_tristan.svm(x_train_ozone, x_test_ozone, y_train_ozone, y_test_ozone)
    else:
        print("Choix non reconnu")



if __name__ == '__main__':
    main()

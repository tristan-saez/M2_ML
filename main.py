import pandas as pd
import algorithms as al


def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    car_seats_bdd = pd.read_csv("data/Carseats.csv")
    ozone_bdd = pd.read_table("data/ozone_complet.txt", sep = ";")

    print("=" * 50, "\nCHOIX ALGORITHME\n"+"=" * 50+"\n1. Arbre de décisions\n2. Forêts aléatoires\n"
                                                    "3. Régression ridge\n4. Régression lasso\n5. SVM")
    choice = int(input("\n>"))

    if choice == 1:
        print("=" * 50, "\nARBRE DE DÉCISION\n" + "=" * 50)
        al.DecisionTree.decision_tree(car_seats_bdd)
    elif choice == 2:
        print("=" * 50, "\nFORÊTS ALÉATOIRES\n" + "=" * 50)
        al.RandomForest.random_forest(car_seats_bdd)
    elif choice == 3:
        print("=" * 50, "\nRÉGRESSION RIDGE\n" + "=" * 50)
        al.RidgeRegressor.ridge_regressor(ozone_bdd)
    elif choice == 4:
        print("=" * 50, "\nRÉGRESSION LASSO\n" + "=" * 50)
        al.LassoRegressor.lasso_regressor(ozone_bdd)
    elif choice == 5:
        print("=" * 50, "\nSVM\n" + "=" * 50)
        al.SVM.svm(ozone_bdd)
    else:
        print("Choix non reconnu")



if __name__ == '__main__':
    main()

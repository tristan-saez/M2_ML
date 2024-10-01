import pandas as pd
from code import *


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
    elif choice == 2:
        print("=" * 50, "\nFORÊTS ALÉATOIRES\n" + "=" * 50)
    elif choice == 3:
        print("=" * 50, "\nRÉGRESSION RIDGE\n" + "=" * 50)
    elif choice == 4:
        print("=" * 50, "\nRÉGRESSION LASSO\n" + "=" * 50)
    elif choice == 5:
        print("=" * 50, "\nSVM\n" + "=" * 50)
    else:
        print("Choix non reconnu")



if __name__ == '__main__':
    main()

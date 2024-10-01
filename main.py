import pandas as pd

from algorithms import RidgeRegressor



def main():
    """

    Mettez votre l'appel de chaque fonction ici.
    Exemple :

    DecisionTree.main(data)
    """
    car_seats_bdd = pd.read_csv("data/Carseats.csv")
    ozone_bdd = pd.read_table("data/ozone_complet.txt", sep = ";")

    RidgeRegressor.ridge_regressor(ozone_bdd)



if __name__ == '__main__':
    main()
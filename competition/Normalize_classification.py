import pandas as pd
import numpy as np
import sklearn.model_selection as skmodel
import matplotlib.pyplot as plt

DEFINITIONS = {}


def PreProcessing(train_path, test_path, sep) -> pd.DataFrame:

    data_train = convertToCSV(train_path, sep)
    data_train = cleanData(data_train)
    data_train['salary'] = data_train.apply(lambda x: 0 if x['Salary'] <= 425 else 1, axis = 1)
    data_train = data_train.drop("Salary", axis=1)
    data_train = normalize(data_train)
    X_train = data_train.drop("salary", axis=1)
    Y_train = data_train["salary"]
    
    data_test = convertToCSV(test_path, sep)
    data_test = cleanData(data_test)
    data_test['salary'] = data_test.apply(lambda x: 0 if x['Salary'] <= 425 else 1, axis = 1)
    data_test = data_test.drop("Salary", axis=1)
    data_test = normalize(data_test)
    X_test = data_test.drop("salary", axis=1)
    Y_test = data_test["salary"]


    return X_train, X_test, Y_train, Y_test


def convertToCSV(file_path, sep) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def cleanData(data) -> pd.DataFrame:
    # Supression de la colonne ID
    data = data.drop('Unnamed: 0', axis=1)

    # Conversion des champs de texte en donnée
    data = data.replace({"A": 1, "N": 0})
    data = data.replace({'E': 1, 'W': 0})

    # Supression des lignes ayant des données NaT ou NaN
    data = data.dropna()

    return data


def normalize(data) -> pd.DataFrame:
    # Normalisation des données
    for column in data.columns:
        data[column] = data[column] / data[column].abs().max()

    return data


def testPreProcessing():
    X_train, X_test, Y_train, Y_test = PreProcessing('Hitters_train.csv', 'Hitters_test.csv',',')



if __name__ == "__main__":
    testPreProcessing()

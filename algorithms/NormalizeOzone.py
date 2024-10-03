import pandas as pd
import sklearn.model_selection as skmodel

DEFINITIONS = {}


def PreProcessingOzone(file_path, sep):

    data = convertToCSVOzone(file_path, sep)
    data = cleanData(data)
    data = normalizeOzone(data)

    X = data.drop("maxO3", axis=1)
    Y = data["maxO3"]

    X_train, X_test, Y_train, Y_test = skmodel.train_test_split(
        X, Y, test_size=0.30, random_state=42)

    return X_train, X_test, Y_train, Y_test


def cleanData(data) -> pd.DataFrame:
    # Supression des colonnes inutiles
    data = data.drop("maxO3v", axis=1)

    # Supression des lignes ayant des données NaT ou NaN
    data = data.dropna()

    return data


def convertToCSVOzone(file_path, sep) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def normalizeOzone(data) -> pd.DataFrame:
    # Normalisation des données
    for column in data.columns:
        data[column] = data[column] / data[column].abs().max()

    return data


def visualizeOzone(data):
    blank


def testPreProcessingOzone():
    X_train, X_test, Y_train, Y_test = PreProcessingOzone(
        'M2_ML/data/ozone_complet.txt', ';')
    print(X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    testPreProcessingOzone()

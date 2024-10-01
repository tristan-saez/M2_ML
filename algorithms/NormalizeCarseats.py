import pandas as pd
import sklearn.model_selection as skmodel

DEFINITIONS = {}


def PreProcessingCarSeats(file_path, sep) -> pd.DataFrame:

    data = convertToCSVCarSeats(file_path, sep)
    data = cleanDataCarSeats(data)
    data = normalizeCarSeats(data)

    X = data.drop("High", axis=1)
    Y = data["High"]

    X_train, X_test, Y_train, Y_test = skmodel.train_test_split(
        X, Y, test_size=0.30, random_state=42)

    return X_train, X_test, Y_train, Y_test

    return data


def convertToCSVCarSeats(file_path, sep) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def cleanDataCarSeats(data) -> pd.DataFrame:
    # Supression de la colonne ID
    data = data.drop('Unnamed: 0', axis=1)

    # Conversion des champs de texte en donnée
    textColumns = ['ShelveLoc', 'Urban', 'US', 'High']

    for column in textColumns:
        labels, level = pd.factorize(data[column])
        data[column] = labels

    return data


def normalizeCarSeats(data) -> pd.DataFrame:
    # Normalisation des données
    for column in data.columns:
        data[column] = data[column] / data[column].abs().max()

    return data


def testPreProcessingCarSeats():
    print(PreProcessingCarSeats('M2_ML/data/Carseats.csv', ','))


if __name__ == "__main__":
    testPreProcessingCarSeats()

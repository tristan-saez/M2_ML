import pandas as pd
import sklearn.model_selection as skmodel
import matplotlib.pyplot as plt

DEFINITIONS = {}


def PreProcessingOzone(file_path, sep):

    data = convertToCSVOzone(file_path, sep)
    data = cleanDataOzone(data)

    X = data.drop("maxO3", axis=1)
    X = normalizeOzone(X)
    Y = data["maxO3"]

    X_train, X_test, Y_train, Y_test = skmodel.train_test_split(
        X, Y, test_size=0.20, random_state=42)

    return X_train, X_test, Y_train, Y_test


def cleanDataOzone(data) -> pd.DataFrame:
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


def visualizeOzone(file_path, sep):
    data = convertToCSVOzone(file_path, sep)
    data = cleanDataOzone(data)
    data = normalizeOzone(data)

    # Définition des types de colonnes
    y = ["T6",
         "T9",
         "T12",
         "T15",
         "T18",
         "Ne6",
         "Ne9",
         "Ne12",
         "Ne15",
         "Ne18",
         "Vdir6",
         "Vvit6",
         "Vdir9",
         "Vvit9",
         "Vdir12",
         "Vvit12",
         "Vdir15",
         "Vvit15",
         "Vdir18",
         "Vvit18",
         "Vx"]
    n_scatter_plot(data, y)


def n_scatter_plot(data, column):
    for column in column:
        data_xaxis = data["maxO3"].to_numpy()
        data_yaxis = data[column].to_numpy()

        fig, ax = plt.subplots()
        ax.set_title('Valeur de MaxO3 en fonction de la valeur de ' + column)
        ax.set_xlabel('Valeur normalisé de MaxO3')
        ax.set_ylabel('Valeur normalisé de ' + column)

        bplot = ax.scatter(data_xaxis, data_yaxis)

        plt.show()


def testPreProcessingOzone():
    X_train, X_test, Y_train, Y_test = PreProcessingOzone(
        'M2_ML/data/ozone_complet.txt', ';')
    print(X_train, X_test, Y_train, Y_test)


def testVisualizeOzone():
    visualizeOzone('M2_ML/data/ozone_complet.txt', ';')


if __name__ == "__main__":
    # testPreProcessingOzone()
    testVisualizeOzone()

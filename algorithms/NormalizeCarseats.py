import pandas as pd
import numpy as np
import sklearn.model_selection as skmodel
import matplotlib.pyplot as plt

DEFINITIONS = {}


def PreProcessingCarSeats(file_path, sep) -> pd.DataFrame:

    data = convertToCSVCarSeats(file_path, sep)
    data = cleanDataCarSeats(data)
    data = normalizeCarSeats(data)

    X = data.drop("High", axis=1)
    Y = data["High"]

    X_train, X_test, Y_train, Y_test = skmodel.train_test_split(
        X, Y, test_size=0.20, random_state=42)

    return X_train, X_test, Y_train, Y_test


def convertToCSVCarSeats(file_path, sep) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=sep)


def cleanDataCarSeats(data) -> pd.DataFrame:
    # Supression de la colonne ID
    data = data.drop('Unnamed: 0', axis=1)

    # Conversion des champs de texte en donnée
    data = pd.get_dummies(data, columns=['ShelveLoc'], drop_first=False)
    data = data.replace({True: 1, False: 0})
    data = data.replace({'Yes': 1, 'No': 0})

    textColumns = ['Urban', 'US', 'High']

    # Supression des lignes ayant des données NaT ou NaN
    data = data.dropna()

    return data


def normalizeCarSeats(data) -> pd.DataFrame:
    # Normalisation des données
    for column in data.columns:
        data[column] = data[column] / data[column].abs().max()

    return data


def visualizeCarSeats(file_path, sep):
    data = convertToCSVCarSeats(file_path, sep)
    data = cleanDataCarSeats(data)
    data = normalizeCarSeats(data)

    # Division des données suivant la valeur de High (Yes or No)
    data_no = data.loc[data['High'] == 0]
    data_yes = data.loc[data['High'] == 1]

    # Définition des types de colonnes
    binary_columns = ['Urban', 'US', 'ShelveLoc_Bad',
                      'ShelveLoc_Medium', 'ShelveLoc_Good']
    non_binary_columns = ['CompPrice', 'Income',
                          'Advertising', 'Population', 'Price', 'Age', 'Education']

    binary_plot(data_no, data_yes, binary_columns)
    non_binary_plot(data_no, data_yes, non_binary_columns)


def binary_plot(data_no, data_yes, binary_columns):
    for column in binary_columns:
        count_no = data_no[column].value_counts()
        count_yes = data_yes[column].value_counts()

        labels = ('Low', 'High')
        values = {'No': (count_no[0], count_no[1]),
                  'Yes': (count_yes[0], count_yes[1])}

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for label, count in values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, count, width, label=label)
            ax.bar_label(rects)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Nombre d''occurences de ' + column)
        ax.set_title('Valeur de High en fonction de la valeur de ' + column)
        ax.set_xticks(x + width/2, labels)
        ax.legend(loc='upper left', ncols=3)
        ax.set_ylim(0, 250)

        plt.show()


def non_binary_plot(data_no, data_yes, non_binary_column):
    for column in non_binary_column:
        y_no = data_no[column].to_numpy()
        y_yes = data_yes[column].to_numpy()

        labels = ['Low', 'High']

        fig, ax = plt.subplots()
        ax.set_title('Valeur de High en fonction de la valeur de ' + column)
        ax.set_ylabel('Valeur normalisé de ' + column)

        bplot = ax.boxplot([y_no, y_yes],
                           labels=labels)  # will be used to label x-ticks

        plt.show()


def testPreProcessingCarSeats():
    print(PreProcessingCarSeats('M2_ML/data/Carseats.csv', ','))


def testVisualizeCarSeats():
    visualizeCarSeats('M2_ML/data/Carseats.csv', ',')


if __name__ == "__main__":
    # testPreProcessingCarSeats()
    testVisualizeCarSeats()

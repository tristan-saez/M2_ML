from sklearn import metrics as mt
from matplotlib import pyplot as plt

def check_score(type, y_true, y_pred):
    """

    :param type: class of algorithm (classifier / regressor)
    :param y_true: true labels of dataset
    :param y_pred: predicted labels of dataset
    """
    if type == "classifier":
        classifier_scoring(y_true, y_pred)
        conf_mat = get_conf_matrix(y_true, y_pred)
        show_conf_matrix(conf_mat)
    elif type == "regressor":
        regressor_scoring(y_true, y_pred)


def classifier_scoring(y_true, y_pred):
    print("["+"="*20+" CLASSIFIER SCORING "+"="*20+"]\n")

    accuracy = mt.accuracy_score(y_true, y_pred)
    print(f"accuracy : {accuracy*100}%")
    f1_score = mt.f1_score(y_true, y_pred)
    print(f"f1 score : {f1_score:.4f}")
    recall = mt.recall_score(y_true, y_pred)
    print(f"recall : {recall*100}%")

    print("\n[" + "=" * 20 + " CLASSIFIER SCORING " + "=" * 20 + "]")

def regressor_scoring(y_true, y_pred):
    print("[" + "=" * 20 + " REGRESSOR SCORING " + "=" * 20 + "]\n")

    max_err = mt.max_error(y_true, y_pred)
    print(f"max error : {max_err:.4f}")
    rms_err = mt.root_mean_squared_error(y_true, y_pred)
    print(f"Root Mean Squared error : {rms_err:.4f}")
    r2 = mt.r2_score(y_true, y_pred)
    print(f"r2 score : {r2:.4f}")

    print("\n[" + "=" * 20 + " REGRESSOR SCORING " + "=" * 20 + "]")


def get_conf_matrix(y_true, y_pred):
    print("calculating confusion matrix...\n")
    conf_mat = mt.confusion_matrix(y_true, y_pred)

    return conf_mat


def show_conf_matrix(conf_mat):
    disp = mt.ConfusionMatrixDisplay(conf_mat, display_labels=True)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    check_score("classifier", [1, 1, 1, 1], [0, 1, 1, 1])
    check_score("regressor", [1, 1, 1, 1], [0, 1, 1, 1])
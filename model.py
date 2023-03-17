import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def Test_model(model, data, result):
    """
    this function will test different model
    """

    ratio_train_test = 4000

    X_train = data[:ratio_train_test]
    X_test = data[ratio_train_test:]
    y_train = result[:ratio_train_test]
    y_test = result[ratio_train_test:]

    model.fit(X_train, y_train)
    predict_data = model.predict(X_test)

    # cross validation
    cross_score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    print("Cross validation score:")
    print(cross_score)
    print()

    # Confusion matrix
    matrix = confusion_matrix(y_test, predict_data)
    print("Confusion matrix:")
    print(matrix)
    print()

    # Recall
    recall = recall_score(y_test, predict_data)
    print("Recall score:")
    print(recall)
    print()

    # Accuracy
    precision = precision_score(y_test, predict_data)
    print("Precision score:")
    print(precision)
    print()

    # f1 score
    f1 = f1_score(y_test, predict_data)
    print("F1 score:")
    print(f1)
    print()

    # courbe de roc
    fpr, tpr, thresholds = roc_curve(y_test, predict_data)
    return (fpr, tpr)
    # plot_roc_curve(fpr, tpr)
    # plt.show()


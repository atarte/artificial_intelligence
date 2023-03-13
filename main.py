import utils as utl
import pipeline as pl
import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

if __name__ == "__main__":

    # Get data set directly transform from the pipeline
    data, result = pl.TransformData()

    # Get a test data set to train our model
    X_train, X_test, y_train, y_test = data[:2000], data[2000:], result[:2000], result[2000:]
    # train_set = train_test_split(data)
    # test_result = train_test_split(result)

    # Train the model with test data
    sgd_clf = SGDClassifier(max_iter=5)
    sgd_clf.fit(X_train, y_train)
    # clf = MLPClassifier(random_state=1, max_iter=300)
    # clf.fit(test_set, test_result)
    predict_data = sgd_clf.predict(X_test)

    # cross validation
    cross_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    print(cross_score)

    # Confusion matrix
    matrix = confusion_matrix(y_test, predict_data)
    print(matrix)

    # Recall
    recall = recall_score(y_test, predict_data)
    print(recall)

    # Accuracy
    precision = precision_score(y_test, predict_data)
    print(precision)

    # f1 score
    f1 = f1_score(y_test, predict_data)
    print(f1)

    # courbe de roc
    fpr, tpr, thresholds = roc_curve(y_test, predict_data)
    plot_roc_curve(fpr, tpr)
    plt.show()


    # Predict the data with the model
    # prediction = clf.predict(data)
    # print(prediction)

    # score = clf.score(data, result)
    # print(score)bloop
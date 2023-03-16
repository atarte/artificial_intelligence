import utils as utl
import model as mdl
import pipeline as pl
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)


if __name__ == "__main__":

    # Get data set directly transform from the pipeline
    data, result = pl.TransformData()

    model = SGDClassifier(max_iter=5)

    models_name = ['SGD', 'Logistic Reg', 'Arbre', 'RandomForestClassifier', 'MLP']
    models_list = [SGDClassifier(max_iter=5), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=10), MLPClassifier(max_iter=5)]

    roc_array = []

    for i in range(len(models_list)):
        print(models_name[i])
        roc_data = mdl.Test_model(models_list[i], data, result)
        print()

        roc_array.append(roc_data)
    
    len_roc = len(roc_array)
    for i in range(len_roc-1):
        plt.plot(roc_array[i][0], roc_array[i][1], linewidth=2, label=models_name[i])

    mdl.plot_roc_curve(roc_array[len_roc-1][0], roc_array[len_roc-1][1], models_name[len_roc-1])
    plt.legend(loc="lower right", fontsize=16)
    plt.show()

    # Get a test data set to train our model
    # X_train, X_test, y_train, y_test = data[:2000], data[2000:], result[:2000], result[2000:]
    # train_set = train_test_split(data)
    # test_result = train_test_split(result)

    # Train the model with test data
    # model = SGDClassifier(max_iter=5)
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # # clf = MLPClassifier(random_state=1, max_iter=300)
    # # clf.fit(test_set, test_result)
    # predict_data = model.predict(X_test)

    # cross validation
    # cross_score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    # print(cross_score)

    # # Confusion matrix
    # matrix = confusion_matrix(y_test, predict_data)
    # print(matrix)

    # # Recall
    # recall = recall_score(y_test, predict_data)
    # print(recall)

    # # Accuracy
    # precision = precision_score(y_test, predict_data)
    # print(precision)

    # # f1 score
    # f1 = f1_score(y_test, predict_data)
    # print(f1)

    # # courbe de roc
    # fpr, tpr, thresholds = roc_curve(y_test, predict_data)
    # plot_roc_curve(fpr, tpr)
    # plt.show()


    # Predict the data with the model
    # prediction = clf.predict(data)
    # print(prediction)

    # score = clf.score(data, result)
    # print(score)bloop
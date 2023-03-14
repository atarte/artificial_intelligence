import utils as utl
import model as mdl
import pipeline as pl
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":

    # Get data set directly transform from the pipeline
    data, result = pl.TransformData()

    model = SGDClassifier(max_iter=5)

    mdl.Test_model(model, data, result)
    
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
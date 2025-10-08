'''
Machine Learning Classifiers: KNN and SVC for シ & ツ characters
This script loads an image dataset, preprocesses the images, splits the data into training and testing
sets, trains KNN and SVC classifiers, evaluates their performance, and visualizes the results.
'''
from Preprocessing.load import load_img
from Preprocessing.spliting import split_data

from Training.knn.knn import train_knn
from Training.svc.svc import train_svc
from Training.eval import evaluate_model
from Plot.plot import plot_results
from Config.config import DATASET_PATH  

def main():
    images, labels = load_img(DATASET_PATH)
    X_train, X_test, y_train, y_test = split_data(images, labels)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    knn_model = train_knn(X_train, y_train)
    svc_model = train_svc(X_train, X_test, y_train, y_test)

    print("KNN Model Evaluation:")
    y_pred_knn = evaluate_model(knn_model, X_test, y_test)
    # plot_results(y_test, y_pred_knn, "KNN Confusion Matrix")
    print(y_test, y_pred_knn, "KNN Confusion Matrix")

    print("SVC Model Evaluation:")
    y_pred_svc = evaluate_model(svc_model, X_test, y_test)
    # plot_results(y_test, y_pred_svc, "SVC Confusion Matrix")
    print(y_test, y_pred_svc, "SVC Confusion Matrix")

if __name__ == "__main__":
    main()

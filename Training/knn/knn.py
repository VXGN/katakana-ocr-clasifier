from sklearn.neighbors import KNeighborsClassifier

from Config.config import KNN_NEIGHBORS


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    knn.fit(X_train, y_train)
    return knn
from sklearn.model_selection import train_test_split

from Config.config import RANDOM_STATE, TEST_SIZE

def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)
    return X_train, X_test, y_train, y_test
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from Config.config import RANDOM_STATE, SVC_KERNEL
from Plot.plot import plot_svc_confusion_grid

def train_svc(X_train, X_test, y_train, y_test, C_values=[0.1, 1, 2], kernels = SVC_KERNEL):
    best_svc_model = None
    best_svc_score = 0
    results = []
    for C in C_values:
        for kernel in kernels:
            svc_model = SVC(C=C, kernel=kernel, random_state=RANDOM_STATE)
            svc_model.fit(X_train, y_train)
            score = svc_model.score(X_test, y_test)
            y_pred = svc_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            results.append((f"C={C}, kernel={kernel}", cm))
            if score > best_svc_score:
                best_svc_score = score
                best_svc_model = svc_model
            print(f"Best SVC Score: {best_svc_score}")
    plot_svc_confusion_grid(results)
    return best_svc_model
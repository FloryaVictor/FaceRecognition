from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class Detector:
    def __init__(self):
        self.svm = None
        self.face_shape = None

    def train(self, X_train, Y_train):
        if X_train.size > 0:
            self.face_shape = X_train[0].shape
        grid = GridSearchCV(LinearSVC(), {'C': [0.1, 1.0, 2.0, 4.0, 8.0]})
        grid.fit(X_train, Y_train)
        self.svm = grid.best_estimator_
        self.svm.fit(X_train, Y_train)

    def detect(self, img):
        pass
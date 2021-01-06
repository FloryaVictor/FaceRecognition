from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from skimage import feature
from skimage import transform
import numpy as np


class Detector:
    def __init__(self):
        self.svm = None
        self.face_shape = None

    def train(self, X_train: np.ndarray, Y_train: np.ndarray):
        if X_train.size > 0:
            self.face_shape = X_train[0].shape
        else:
            self.face_shape = (0, 0)
        X_train = np.array([feature.hog(img) for img in X_train])
        grid = GridSearchCV(LinearSVC(), {'C': [0.1, 1.0, 2.0, 4.0, 8.0]})
        grid.fit(X_train, Y_train)
        self.svm = grid.best_estimator_
        self.svm.fit(X_train, Y_train)

    def _get_patches(self, img, patch_size=None, dx=2, dy=2):
        scale = False
        if patch_size is None:
            patch_size = self.face_shape
        if patch_size != self.face_shape:
            scale = True
        Ni, Nj = patch_size[0], patch_size[1]
        for i in range(0, img.shape[0] - Ni, dx):
            for j in range(0, img.shape[1] - Ni, dy):
                patch = img[i:i + Ni, j:j + Nj]
                if scale:
                    patch = transform.resize(patch, self.face_shape)
                yield (i, j), patch

    def detect(self, img, shape=None):
        if shape is None:
            shape = self.face_shape
        indices, patches = zip(*self._get_patches(img, shape))
        faces = []
        for k in range(len(patches)):
            if self.svm.predict(feature.hog(patches[k])).sum():
                i, j = indices[k][0], indices[k][1]
                faces.append([i, j, shape[0], shape[1]])
        return faces

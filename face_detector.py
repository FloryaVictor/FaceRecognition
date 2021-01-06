from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from skimage import feature
from skimage import transform
import numpy as np
import pickle
import cv2


class Detector:
    """Detector can be trained on a set of facial and non-facial images to
     find faces on other images"""
    def __init__(self):
        self.svm = None
        self.face_shape = None

    def train(self, X_train: np.ndarray, Y_train: np.ndarray):
        """Train Detector object on a set of examples
            :param X_train: Set of images
            :param Y_train: Set of labels for images
        """
        if X_train.size > 0:
            self.face_shape = X_train[0].shape
        else:
            self.face_shape = (0, 0)
        X_train = np.array([feature.hog(img) for img in X_train])
        grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
        grid.fit(X_train, Y_train)
        self.svm = grid.best_estimator_
        self.svm.fit(X_train, Y_train)

    def _get_patches(self, img, patch_size=None, dx=2, dy=2):
        """Turns image into sequence of patches"""
        scale = False
        if patch_size is None:
            patch_size = self.face_shape
        elif patch_size != self.face_shape:
            scale = True
        Ni, Nj = patch_size[0], patch_size[1]
        for i in range(0, img.shape[0] - Ni, dx):
            for j in range(0, img.shape[1] - Ni, dy):
                patch = img[i:i + Ni, j:j + Nj]
                if scale:
                    patch = transform.resize(patch, self.face_shape)
                yield (i, j), patch

    def detect(self, img, shape=None):
        """Detects faces on an image
        :param img: image for search
        :param shape: shape of face to be found, face_shape by default
        """
        if shape is None:
            shape = self.face_shape
        indices, patches = zip(*self._get_patches(img, shape))
        faces = []
        labels = self.svm.predict([feature.hog(patch) for patch in patches])
        for k in range(len(labels)):
            if labels[k]:
                i, j = indices[k][0], indices[k][1]
                faces.append([i, j, i + shape[0], j + shape[1]])
        faces = cv2.groupRectangles(faces, 1, eps=0.5)[0]
        return faces

    def save(self, path: str):
        """Saves Detector object to file"""
        with open(path, 'wb') as fid:
            pickle.dump(self, fid)

    def load(self, path: str):
        """Loads Detector object from file"""
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            self.svm = obj.svm
            self.face_shape = obj.face_shape


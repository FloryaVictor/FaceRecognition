import face_detector

from sklearn.datasets import fetch_lfw_people
from skimage import data, transform
from sklearn.feature_extraction.image import PatchExtractor
from skimage import data, color, feature
import skimage.data
import numpy as np
from itertools import chain

faces = fetch_lfw_people()
positive_patches = faces.images
imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]


def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches


negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])


X_train = np.array([feature.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])

y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
d = face_detector.Detector()
d.train(X_train, y_train)
print(d.svm)
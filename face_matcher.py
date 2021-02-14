import cv2
import os
import numpy as np
import dlib


class Matcher:
    CNN_MODEL_PATH = 'models/cnn_face_matching.dat'
    FACE_SHAPE = (150, 150)
    THRESHOLD = 0.5

    def __init__(self, known_faces_path):
        """
        :param known_faces_path: directory with known photos
        """
        self.known_encodings = []
        self.known_names = []
        self.model = dlib.face_recognition_model_v1(self.CNN_MODEL_PATH)
        self.shape_predictor = dlib.shape_predictor()
        for img_name in os.listdir(known_faces_path):
            img = cv2.cvtColor(cv2.imread(os.path.join(known_faces_path, img_name)),
                               cv2.COLOR_BGR2RGB)
            name = img_name.split('_')
            name.pop()
            name = ' '.join(name)
            encoding = self._get_encoding(img)
            self.known_encodings.append(encoding)
            self.known_names.append(name)

    def _get_encoding(self, img):
        resized = cv2.resize(img, self.FACE_SHAPE)
        return np.array(self.model.compute_face_descriptor(resized))

    def match(self, img):
        unknown_encoding = self._get_encoding(img)
        encodings_sub = []
        for i in range(len(self.known_encodings)):
            encodings_sub.append(self.known_encodings[i] - unknown_encoding)
        distances = np.linalg.norm(encodings_sub, axis=1)
        matches = list(distances <= self.THRESHOLD)
        if not matches:
            return ""
        count = {}
        for i in [i for (i, m) in enumerate(matches) if m]:
            name = self.known_names[i]
            count[name] = count.get(name, 0) + 1
        name = max(count, key=count.get)
        return name

import dlib


class Detector:
    """Detector uses pretrained model from dlib for face detection"""
    CNN_MODEL_PATH = 'models/cnn_face_detection.dat'

    def __init__(self, model_type='hog'):
        """
        :param model_type: type of the ml model: cnn or hog-based detector
        """
        if model_type == 'hog':
            self.model = dlib.get_frontal_face_detector()
        elif model_type == 'cnn':
            self.model = dlib.cnn_face_detection_model_v1(self.CNN_MODEL_PATH)
        else:
            raise ValueError('Wrong model type')

    def detect(self, img, scale=1.0):
        """Detects faces on an image
        :param img: image for search
        :param scale: scale of search window,0.0 < scale <= 1.0
        """
        if scale == 0:
            raise ValueError("Zero scale is not accepted")
        scale = round(1 / scale)
        faces = self.model(img, scale)
        return faces

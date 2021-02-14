import sys
import os
import ntpath
import uuid
import cv2
import face_detector


def get_suffix():
    return uuid.uuid4().hex[:6].upper()


detector = face_detector.Detector()


for filename in sys.argv[1:]:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    name, extension = os.path.splitext(filename)
    name = os.path.basename(name)
    name = "known_faces/" + name + "_" + get_suffix() + extension
    face = detector.detect(gray)
    if not face:
        continue
    face = face[0]
    x = face.left()
    y = face.top()
    w = face.right() - face.left()
    h = face.bottom() - face.top()
    cv2.imwrite(name, img[y: y + h, x: x + w])


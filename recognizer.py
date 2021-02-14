import sys
import cv2
import face_detector
import face_matcher

KNOWN_FACES_DIR = 'known_faces'


def main(argv):
    detector = face_detector.Detector()
    matcher = face_matcher.Matcher(KNOWN_FACES_DIR)
    for filename in argv:
        img = cv2.imread(filename)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detect(gray_img)
        if not face:
            continue
        face = face[0]
        x = face.left()
        y = face.top()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        name = matcher.match(rgb_img[y: y + h, x: x + w])
        cv2.rectangle(img, (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)
        text_y = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(img, name, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        cv2.imshow(filename, img)
        while 1:
            if cv2.waitKey(33) != -1:
                break


if __name__ == '__main__':
    main(sys.argv[1:])

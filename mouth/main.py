import cv2
import dlib
import numpy as np

PREDICOTR_PATH = "./shape_predictor_68_face_landmarks.dat"

def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,
            color=(1, 2, 255))
        cv2.circle(img, pos, 3, color=(0, 2, 2))
    return img

def center(pts):
    lip_mean = np.mean(pts, axis=0)
    return int(lip_mean[:, 1])


def detect_speaking(img):
    rects = detector(img, 1)
    if len(rects) != 1:
        return img, 0
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

    img_with_landmarks = annotate_landmarks(img, landmarks)

    top_lip_center = center(landmarks[50:53] + landmarks[61:64])
    bottom_lip_center = center(landmarks[65:68] + landmarks[56:59])
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return img_with_landmarks, lip_distance


predictor = dlib.shape_predictor(PREDICOTR_PATH)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
yawn_status = False
lip_distance_list = list()

while True:
    ret, frame = cap.read()
    img_with_landmarks, lip_distance = detect_speaking(frame)
    # lip_distance_list = lip_distance[:19] + [lip_distance] # Not Use right now.

    # TODO. replace constant to dynamic value calculated from mouth size.
    if lip_distance > 45:
        yawn_status = True        
        cv2.putText(frame, "Please, close the mouth for COVID-19", (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        yawn_status = False

    # cv2.imshow("Live landmarks", img_with_landmarks)
    cv2.imshow("Yawn status", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
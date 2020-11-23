# code from https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

from __future__ import print_function
import cv2 as cv
import argparse
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
    eyes = eyes_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in eyes:
        center = (x + w//2, y + h//2)
        radius = int(round((w + h)*0.25))
        frame = cv.circle(frame, center, radius, (255, 0, 0 ), 4)
    noses = nose_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in noses:
        center = (x + w//2, y + h//2)
        radius = int(round((w + h)*0.25))
        frame = cv.circle(frame, center, radius, (255, 255, 0), 4)
    mouths = mouth_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in mouths:
        center = (x + w//2, y + h//2)
        radius = int(round((w + h)*0.25))
        frame = cv.circle(frame, center, radius, (0, 255, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--mouth_cascade', help='Path to mouth cascade.', default='data/haarcascades/haarcascade_mcs_mouth.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.', default='data/haarcascades/Nariz.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
mouth_cascade_name = args.mouth_cascade
nose_cascade_name = args.nose_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()
nose_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)
if not mouth_cascade.load(cv.samples.findFile(mouth_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)
if not nose_cascade.load(cv.samples.findFile(nose_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device, cv.CAP_DSHOW)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
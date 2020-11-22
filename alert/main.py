# code partly from https://livecodestream.dev/post/2020-07-03-detecting-face-features-with-python/
#              and https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
import cv2
import dlib
import argparse
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector() # pylint: disable=no-member

# Load the predictor
predictor = dlib.shape_predictor("../landmark/shape_predictor_68_face_landmarks.dat") # pylint: disable=no-member

# read the image
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='../cascade/data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../cascade/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--mouth_cascade', help='Path to mouth cascade.', default='../cascade/data/haarcascades/haarcascade_mcs_mouth.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.', default='../cascade/data/haarcascades/Nariz.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
mouth_cascade_name = args.mouth_cascade
nose_cascade_name = args.nose_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
mouth_cascade = cv2.CascadeClassifier()
nose_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)
if not mouth_cascade.load(cv2.samples.findFile(mouth_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)
if not nose_cascade.load(cv2.samples.findFile(nose_cascade_name)): # pylint: disable=maybe-no-member
    print('--(!)Error loading eyes cascade')
    exit(0)

while True:
    _, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    # Convert image into grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)
    
    proper = False
    nose = mouth = False

    # Use detector to find landmarks
    faces = detector(frame_gray)
    
    if len(faces) != 0:
        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(image=frame_gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        mouth = nose = True

    # face not found through detector
    else:
        faces = face_cascade.detectMultiScale(frame_gray)
        # face found through cascade
        if len(faces)>=1:
            for (x,y,w,h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
                faceROI = frame_gray[y:y+h,x:x+w]

                eyeLowerCenter = -1
                #-- In each face, detect eyes
                eyes = eyes_cascade.detectMultiScale(faceROI)                
                for (x2,y2,w2,h2) in eyes:
                    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                    radius = int(round((w2 + h2)*0.25))
                    frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
                
                    lowestEye = max(eyes, key=lambda x:x[1])
                    eyeLowerCenter = lowestEye[1]+lowestEye[3]

                cnt = 0
                noses = nose_cascade.detectMultiScale(faceROI)
                for (x2,y2,w2,h2) in noses:
                    if y2+h2/2 < eyeLowerCenter:
                        continue
                    cnt += 1
                    nose = True
                    nose_center = (x + x2 + w2//2, y + y2 + h2//2)
                    radius = int(round((w2 + h2)*0.25))
                    frame = cv2.circle(frame, nose_center, radius, (255, 255, 0), 4)
                
                mouths = mouth_cascade.detectMultiScale(faceROI)
                for (x2,y2,w2,h2) in mouths:
                    if y2 < eyeLowerCenter:
                        continue
                    cnt += 1
                    mouth=True
                    mouth_center = (x + x2 + w2//2, y + y2 + h2//2)
                    radius = int(round((w2 + h2)*0.25))
                    frame = cv2.circle(frame, mouth_center, radius, (0, 255, 0), 4)
                
                if cnt == 0:
                    proper = True
                    break

        # face not found through cascade
        else:
            eyeLowerCenter = -1

            eyes = eyes_cascade.detectMultiScale(frame_gray)

            # eye found
            if len(eyes) != 0:

                sumCenter = [0, 0]
                oneEye = eyes[0]
                for (x,y,w,h) in eyes:
                    center = (x + w//2, y + h//2)
                    radius = int(round((w + h)*0.25))
                    frame = cv2.circle(frame, center, radius, (255, 0, 0), 4)

                    if eyeLowerCenter < center[1]:
                        eyeLowerCenter = center[1] 
                    
                    sumCenter[0] += center[0]
                    sumCenter[1] += center[1]

                cnt = 0
                validNose = None
                minSqDist = 10**8
                midEye = None
                subEye = None
                distSqEye = None

                # to check if detected noses and mouths are in valid position
                if len(eyes)==2:
                    midEye = [sumCenter[0]/2, sumCenter[1]/2]
                    subEye = np.array([eyes[1][0]-eyes[0][0], eyes[1][1]-eyes[0][1]])
                    distSqEye = np.sum(subEye*subEye)
                    subEye = subEye/np.linalg.norm(subEye)
                
                lowerROI = frame_gray[eyeLowerCenter:,:]
                noses = nose_cascade.detectMultiScale(lowerROI)
                for (x2,y2,w2,h2) in noses:
                    if (x2-x)**2 + y2**2 < minSqDist:
                        validNose = (x2,y2,w2,h2)
                if validNose != None:
                    x2,y2,w2,h2 = validNose
                    nose_center = (x2 + w2//2, y + y2 + h2//2)
                    if midEye != None:
                        toMidEye = np.array([midEye[0]-nose_center[0],midEye[1]-nose_center[1]])
                        toMidEye = toMidEye/np.linalg.norm(toMidEye)
                        degree = np.arccos(np.clip(subEye.dot(toMidEye),-1.0,1.0)*180/np.pi/90)
                        # print('nose',degree)
                        if abs(degree-90)<30:
                            cnt += 1
                            nose = True
                            radius = int(round((w2 + h2)*0.25))
                            frame = cv2.circle(frame, nose_center, radius, (255, 255, 0), 4)
                    else:
                        cnt += 1
                        nose = True
                        radius = int(round((w2 + h2)*0.25))
                        frame = cv2.circle(frame, nose_center, radius, (255, 255, 0), 4)

                mouths = mouth_cascade.detectMultiScale(lowerROI)
                for (x2,y2,w2,h2) in mouths:
                    if validNose != None:
                        if (x2-validNose[0])**2+(y2-validNose[1])**2 > 2*minSqDist:
                            continue
                    mouth_center = (x2 + w2//2, y + y2 + h2//2)

                    if midEye != None:
                        toMidEye = np.array([midEye[0]-mouth_center[0],midEye[1]-mouth_center[1]])
                        toMidEye = toMidEye/np.linalg.norm(toMidEye)
                        degree = np.arccos(np.clip(subEye.dot(toMidEye),-1.0,1.0)*180/np.pi/90)
                        distOneEye = (mouth_center[0]-oneEye[0])**2+(mouth_center[1]-oneEye[1])**2
                        # print('mouth',degree,'/',distOneEye,distSqEye)
                        if abs(degree-90)<30 and distOneEye>distSqEye:
                            cnt += 1
                            mouth = True
                            radius = int(round((w2 + h2)*0.25))
                            frame = cv2.circle(frame, mouth_center, radius, (0, 255, 0), 4)
                    else:
                        cnt += 1
                        mouth = True
                        radius = int(round((w2 + h2)*0.25))
                        frame = cv2.circle(frame, mouth_center, radius, (0, 255, 0), 4)

                if cnt == 0:
                    proper = True

            # eye not found
            else:
                proper = True
                # noses = nose_cascade.detectMultiScale(frame_gray)
                # for (x,y,w,h) in noses:
                #     if y < eyeLowerCenter:
                #         continue
                #     nose = True
                #     center = (x + w//2, y + h//2)
                #     radius = int(round((w + h)*0.25))
                #     frame = cv2.circle(frame, center, radius, (255, 255, 0), 4)
                # mouths = mouth_cascade.detectMultiScale(frame_gray)
                # for (x,y,w,h) in mouths:
                #     if y - h/2 < eyeLowerCenter:
                #         continue
                #     mouth = True
                #     center = (x + w//2, y + h//2)
                #     radius = int(round((w + h)*0.25))
                #     frame = cv2.circle(frame, center, radius, (0, 255, 0), 4)

    strResult = 'Alert!'
    textColor = (0,0,255)
    detected = ''
    if proper:
        strResult = 'Good'
        textColor = (0,255,0)
    # else:    
    #     if nose:
    #         detected += 'nose '
    #     elif mouth:
    #         detected += 'mouth '
    #     if detected != '':
    #         detected += 'detected'

    cv2.putText(frame, strResult, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, textColor, 2)
    # if not proper:
    #     cv2.putText(frame, detected, (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (71,99,255), 2)

    # show the image
    cv2.imshow("Mask Alert", frame)

    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()

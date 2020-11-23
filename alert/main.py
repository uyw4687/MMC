'''
code partly from https://livecodestream.dev/post/2020-07-03-detecting-face-features-with-python/
             and https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
'''
import sys
import cv2
import dlib
import numpy as np

CASC_DIR = '../cascade/'

# valid distance to a mouth relative to the nose from an eye
MOUTH_POS_REL_NOSE_FROM_EYE_COEF = 2

def get_degree(vec1, vec2):
    ''' get degree '''
    return np.arccos(np.clip(vec1.dot(vec2),-1.0,1.0))*180/np.pi

def show_result(frame,proper):
    ''' show result '''
    str_result = 'Alert!'
    text_color = (0,0,255)
    # detected = ''
    if proper:
        str_result = 'Good'
        text_color = (0,255,0)
    # else:
    #     if nose:
    #         detected += 'nose '
    #     elif mouth:
    #         detected += 'mouth '
    #     if detected != '':
    #         detected += 'detected'

    cv2.putText(frame, str_result, (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 2)
    # if not proper:
    #     cv2.putText(frame, detected, (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (71,99,255), 2)

    # show the image
    cv2.imshow("Mask Alert", frame)

    # Exit when escape is pressed
    stop = (cv2.waitKey(delay=1) == 27)
    return stop

def run_landmark(detector, predictor, nose_cascade, frame, frame_gray): # pylint: disable=too-many-locals
    ''' run landmark '''
    # Use detector to find landmarks
    faces = detector(frame_gray)

    proper = True
    # face found
    if len(faces) != 0:
        for face in faces:
            f_x = face.left()  # left point
            f_y = face.top()  # top point
            f_w = face.right()-f_x  # right point
            f_h = face.bottom()-f_y  # bottom point
            # Create landmark object
            landmarks = predictor(image=frame_gray, box=face)
            # Loop through all the points
            for num in range(0, 68):
                pos = (landmarks.part(num).x,landmarks.part(num).y)
                # Draw a circle
                if num==30:
                    cv2.circle(img=frame, center=pos, radius=3, color=(0, 0, 255), thickness=-1)
                else:
                    cv2.circle(img=frame, center=pos, radius=3, color=(0, 255, 0), thickness=-1)

            face_center = np.array([f_x + f_w//2, f_y + f_h//2])
            frame = cv2.ellipse(frame,tuple(face_center),(f_w//2,f_h//2),0,0,360,(255,0,255),4)
            face_roi = frame_gray[f_y:f_y+f_h,f_x:f_x+f_w]

            noses = nose_cascade.detectMultiScale(face_roi)
            for (in_x,in_y,in_w,in_h) in noses:
                nose_center = np.array([f_x + in_x + in_w//2, f_y + in_y + in_h//2])
                radius = int(round((in_w + in_h)*0.25))
                if np.sum((nose_center-face_center)**2) > radius**2:
                    continue
                proper = False
                frame = cv2.circle(frame, tuple(nose_center), radius, (255, 255, 0), 4)

            # if not proper:
            #     mouth = nose = True

    found_face = (len(faces)!=0)

    return found_face,frame,proper

def bad_distance(coo,valid_nose,min_sq_dist):
    ''' bad distance '''
    if valid_nose is not None:
        valid_nose = np.array(valid_nose[:2])
        bad = np.sum((coo-valid_nose)**2) > MOUTH_POS_REL_NOSE_FROM_EYE_COEF*min_sq_dist
        return bad
    return False

def handle_degree(args):
    ''' handle degree '''
    frame,mid_eye,center,sub_eye,in_w,in_h,proper = args
    to_mid_eye = mid_eye-center
    to_mid_eye = to_mid_eye/np.linalg.norm(to_mid_eye)
    degree = get_degree(sub_eye,to_mid_eye)
    # print(degree)
    if abs(degree-90)<30:
        radius = int(round((in_w + in_h)*0.25))
        frame = cv2.circle(frame, tuple(center), radius, (255, 255, 0), 4)
        proper = False
    return frame,proper

def handle_degree_and_dist(args):
    ''' handle degree and distance'''
    frame,mid_eye,center,sub_eye,lower_eye,dist_sq_eye,in_w,in_h,proper = args

    to_mid_eye = mid_eye-center
    to_mid_eye = to_mid_eye/np.linalg.norm(to_mid_eye)
    degree = get_degree(sub_eye,to_mid_eye)
    dist_one_eye = np.sum((center-lower_eye)**2)
    # print(degree)
    if abs(degree-90)<30 and dist_one_eye>dist_sq_eye:
        radius = int(round((in_w + in_h)*0.25))
        frame = cv2.circle(frame, tuple(center), radius, (0, 255, 0), 4)
        proper = False
    return frame,proper

def run_face_cascade(args): # pylint: disable=too-many-locals
    ''' run face cascade '''
    face_cascade,eyes_cascade,nose_cascade,mouth_cascade,frame,frame_gray = args
    proper = True
    faces = face_cascade.detectMultiScale(frame_gray)
    for (f_x,f_y,f_w,f_h) in faces:
        center = (f_x + f_w//2, f_y + f_h//2)
        frame = cv2.ellipse(frame, center, (f_w//2, f_h//2), 0, 0, 360, (255, 0, 255), 4)
        face_roi = frame_gray[f_y:f_y+f_h,f_x:f_x+f_w]

        eye_lower_center = -1
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(face_roi)
        for (in_x,in_y,in_w,in_h) in eyes:
            eye_center = (f_x + in_x + in_w//2, f_y + in_y + in_h//2)
            radius = int(round((in_w + in_h)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

            lowest_eye = max(eyes, key=lambda x:x[1])
            eye_lower_center = lowest_eye[1]+lowest_eye[3]

        noses = nose_cascade.detectMultiScale(face_roi)
        for (in_x,in_y,in_w,in_h) in noses:
            if in_y+in_h/2 < eye_lower_center:
                continue
            proper = False
            # nose = True
            nose_center = (f_x + in_x + in_w//2, f_y + in_y + in_h//2)
            radius = int(round((in_w + in_h)*0.25))
            frame = cv2.circle(frame, nose_center, radius, (255, 255, 0), 4)

        mouths = mouth_cascade.detectMultiScale(face_roi)
        for (in_x,in_y,in_w,in_h) in mouths:
            if in_y < eye_lower_center:
                continue
            proper = False
            # mouth = True
            mouth_center = (f_x + in_x + in_w//2, f_y + in_y + in_h//2)
            radius = int(round((in_w + in_h)*0.25))
            frame = cv2.circle(frame, mouth_center, radius, (0, 255, 0), 4)

        if proper:
            break

    found_face = (len(faces)!=0)

    return found_face,frame,proper

def process_eyes(eyes,frame):
    ''' process_eyes '''
    eye_lower_center = -1
    sum_center = [0, 0]
    lower_eye = None
    np.array(eyes[0][:2])

    for (e_x,e_y,e_w,e_h) in eyes:
        center = (e_x + e_w//2, e_y + e_h//2)
        radius = int(round((e_w + e_h)*0.25))
        frame = cv2.circle(frame, center, radius, (255, 0, 0), 4)

        if eye_lower_center < center[1]:
            eye_lower_center = center[1]
            lower_eye = np.array(center)

        sum_center[0] += center[0]
        sum_center[1] += center[1]

    return eye_lower_center,sum_center,lower_eye,frame

def run_eye_cascade(args): # pylint: disable=too-many-locals
    ''' run eye cascade '''
    eyes_cascade,nose_cascade,mouth_cascade,frame,frame_gray = args
    eyes = eyes_cascade.detectMultiScale(frame_gray)

    if len(eyes)==0:
        found_eye = False
        return found_eye,frame,None

    eye_lower_center,sum_center,lower_eye,frame = process_eyes(eyes,frame)

    proper = True
    valid_nose = None
    min_sq_dist = 10**8
    mid_eye = None
    sub_eye = None
    dist_sq_eye = None

    # to check if detected noses and mouths are in valid position
    if len(eyes)==2:
        mid_eye = np.array([sum_center[0]/2, sum_center[1]/2])
        sub_eye = np.array([eyes[1][0]-eyes[0][0], eyes[1][1]-eyes[0][1]])
        dist_sq_eye = np.sum(sub_eye*sub_eye)
        sub_eye = sub_eye/np.linalg.norm(sub_eye)

    lower_roi = frame_gray[eye_lower_center:,:]
    noses = nose_cascade.detectMultiScale(lower_roi)
    for (in_x,in_y,in_w,in_h) in noses:
        if (in_x-lower_eye[0])**2 + in_y**2 < min_sq_dist:
            valid_nose = (in_x,in_y,in_w,in_h)

    if valid_nose is not None:
        in_x,in_y,in_w,in_h = valid_nose
        nose_center = np.array([in_x + in_w//2, lower_eye[1] + in_y + in_h//2])

        if mid_eye is not None:
            # print('nose')
            args = (frame,mid_eye,nose_center,sub_eye,in_w,in_h,proper)
            frame,proper = handle_degree(args)
            # frame,proper,nose = handle_degree(frame,mid_eye,nose_center,sub_eye)
        else:
            proper = False
            # nose = True
            radius = int(round((in_w + in_h)*0.25))
            frame = cv2.circle(frame, tuple(nose_center), radius, (255, 255, 0), 4)

    mouths = mouth_cascade.detectMultiScale(lower_roi)
    for (in_x,in_y,in_w,in_h) in mouths:

        if bad_distance(np.array([in_x,in_y]),valid_nose,min_sq_dist):
            continue

        mouth_center = np.array([in_x + in_w//2, lower_eye[1] + in_y + in_h//2])

        if mid_eye is not None:
            # print('mouth')
            coo = (in_w,in_h)
            args = (frame,mid_eye,mouth_center,sub_eye,lower_eye,dist_sq_eye,*coo,proper)
            frame,proper = handle_degree_and_dist(args)
            # frame,proper,mouth = handle_degree_and_dist(args)

        else:
            proper = False
            # mouth = True
            radius = int(round((in_w + in_h)*0.25))
            frame = cv2.circle(frame, tuple(mouth_center), radius, (0, 255, 0), 4)

    found_eye = True
    return found_eye,frame,proper

def run(resources):
    ''' run '''
    (detector,predictor), (face_cascade,eyes_cascade,mouth_cascade,nose_cascade), cap = resources
    while True:
        _,frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # Convert image into grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        proper = False
        # nose = mouth = False

        found_face,frame,proper = run_landmark(detector,predictor,nose_cascade,frame,frame_gray)
        # face not found through detector
        if not found_face:
            args = (face_cascade,eyes_cascade,nose_cascade,mouth_cascade,frame,frame_gray)
            found_face,frame,proper = run_face_cascade(args)

            # face not found through cascade
            if not found_face:
                args = (eyes_cascade,nose_cascade,mouth_cascade,frame,frame_gray)
                found_eye,frame,proper = run_eye_cascade(args)

                # eye not found
                if not found_eye:
                    proper = True
                    # noses = nose_cascade.detectMultiScale(frame_gray)
                    # for (x,y,w,h) in noses:
                    #     nose = True
                    #     center = (x + w//2, y + h//2)
                    #     radius = int(round((w + h)*0.25))
                    #     frame = cv2.circle(frame, center, radius, (255, 255, 0), 4)
                    # mouths = mouth_cascade.detectMultiScale(frame_gray)
                    # for (x,y,w,h) in mouths:
                    #     mouth = True
                    #     center = (x + w//2, y + h//2)
                    #     radius = int(round((w + h)*0.25))
                    #     frame = cv2.circle(frame, center, radius, (0, 255, 0), 4)

        if show_result(frame,proper):
            break

def load():
    ''' load '''
    # Load the detector
    detector = dlib.get_frontal_face_detector() # pylint: disable=no-member
    # Load the predictor
    predictor = dlib.shape_predictor("../landmark/shape_predictor_68_face_landmarks.dat") # pylint: disable=no-member
    # Read the image
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    face_cascade_name = CASC_DIR+'data/haarcascades/haarcascade_frontalface_alt.xml'
    eyes_cascade_name = CASC_DIR+'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
    mouth_cascade_name = CASC_DIR+'data/haarcascades/haarcascade_mcs_mouth.xml'
    nose_cascade_name = CASC_DIR+'data/haarcascades/Nariz.xml'
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()
    mouth_cascade = cv2.CascadeClassifier()
    nose_cascade = cv2.CascadeClassifier()

    # Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)): # pylint: disable=maybe-no-member
        print('--(!)Error loading face cascade')
        sys.exit(1)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)): # pylint: disable=maybe-no-member
        print('--(!)Error loading eyes cascade')
        sys.exit(1)
    if not mouth_cascade.load(cv2.samples.findFile(mouth_cascade_name)): # pylint: disable=maybe-no-member
        print('--(!)Error loading eyes cascade')
        sys.exit(1)
    if not nose_cascade.load(cv2.samples.findFile(nose_cascade_name)): # pylint: disable=maybe-no-member
        print('--(!)Error loading eyes cascade')
        sys.exit(1)

    return (detector,predictor), (face_cascade,eyes_cascade,mouth_cascade,nose_cascade), cap

def finish(resources):
    ''' finish '''
    _,_,cap = resources
    # When everything done, release the video capture and video write objects
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()

def main():
    ''' main '''
    resources = load()
    run(resources)
    finish(resources)

if __name__ == "__main__":
    main()

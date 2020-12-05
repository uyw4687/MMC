import cv2
import mask_func

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

CASC_DIR = '../cascade/'
eyes_cascade_name = CASC_DIR+'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
eyes_cascade = cv2.CascadeClassifier()

if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    sys.exit(1)

while True:
    ret, frame = cap.read()
    cv2.imshow('original', frame)
    
    frame_draw = frame.copy()

    # for cnt in mask_func.get_simple_contours(frame):
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 0, 255), 4) # just contour
        
    for cnt in mask_func.get_mask_contours(frame):
        x, y, w, h = cv2.boundingRect(cnt)

        # rollback on dilation effect
        x -= 10
        y -= 10
        w += 20
        h += 20

        top = max(0, y - h)
        eye_range = frame[top:y + h // 2, x:x + w]
        
        eyes = eyes_cascade.detectMultiScale(eye_range)
        if len(eyes) == 0:
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)  # contours that have mask features
        else:
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)  # contours that have mask features with eyes

    cv2.imshow('frame', frame_draw)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

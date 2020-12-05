import cv2
import numpy as np

def get_simple_contours(img):
    img_canny = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 50, 200)
    img_dilation = cv2.dilate(img_canny, None, iterations=10)

    contours, _ = cv2.findContours(255 - img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def get_mask_contours(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def contour_to_mask(cnt):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    # check the contour is white color or black color
    def gray_mask_contour_condition(cnt):
        mask = contour_to_mask(cnt)
        img_sv = img_hsv[:,:, 1].astype('int') * img_hsv[:,:, 2]
        
        sv_values = img_sv[mask == 255]
        count_condition = np.count_nonzero(cv2.inRange(sv_values, 0, 5000))
        count_total = np.count_nonzero(mask)

        return count_condition / count_total > 0.99

    # check the contour is solid color
    def color_mask_contour_condition(cnt):
        mask = contour_to_mask(cnt)
        img_h = img_hsv[:,:, 0]
        h_values = img_h[mask == 255]
        h_values2 = h_values + 128

        return h_values.max() - h_values.min() < 20 or h_values2.max() - h_values2.min() < 20

    # check the contour is like shape of common mask
    def mask_shape_contour_condition(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50:
            return False
        
        L = cv2.arcLength(cnt, True)
        A = cv2.contourArea(cnt)

        # L*L/A = pi*4 [circle], 16 [square], 18 [rect with w:h=2:1]
        return L * L / A < 20

    simple_contours = get_simple_contours(img)

    mask_contours = []

    for cnt in simple_contours:
        if (gray_mask_contour_condition(cnt) or color_mask_contour_condition(cnt)) and mask_shape_contour_condition(cnt):
            mask_contours.append(cnt)

    return mask_contours

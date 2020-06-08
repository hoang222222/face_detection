import dlib
import cv2

#Input
cam = cv2.VideoCapture(0)

# Khai bao su dung ham trong thu vien Dlib
detector = dlib.get_frontal_face_detector()

while True:
    ret_val, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image)
    #Khoanh vung nhan dang duoc guong mat
    for det in dets:
        cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 255, 0), 3)
    cv2.imshow('Results', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()

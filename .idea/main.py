import cv2
import pynput
#load some pre_trained data from opencv
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#for video face detection
"""""
webcam=cv2.VideoCapture(0)

while(True):
    successful_frame_read,frame=webcam.read()
    grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('video face detect',grayscale_img)
    cv2.waitKey(1)

    # detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # draw rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show image
    cv2.imshow('face detection ', frame)
    cv2.waitKey(1)


"""""


#choose an image to detect face
img=cv2.imread('grouping.png')

#convert to gray scale
grayscale_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect face
face_coordinates=trained_face_data.detectMultiScale(grayscale_img)

#draw rectangle
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y) ,(x+w ,y+h) , (0,255,0) ,2)

#show image
cv2.imshow('face detection ', img)
cv2.waitKey()



print("code completion")

"""
Algorithm:
Haar feature actually looks for light refle3cted on your face
forehead:light
eyes:dark
checks:light

The thousands of combinations of these gives us the right value

step1: Start with training data 
step2 :Test every haar features thoughout every pixcel on the image
step3 :After it passes it gives a number (diff b/t light and dark)
step4 :cascade(chain of haar features)

"""

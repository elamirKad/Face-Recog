import dlib
from skimage import io
from scipy.spatial import distance
import cv2
import numpy as np
import os
import sys

cap = cv2.VideoCapture(1)

def importPic():
    onlyfiles = next(os.walk("./pictures"))
    for i in onlyfiles:
        print(i)
importPic()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

img = io.imread('elamir.jpg')

win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)


dets = detector(img, 1)


for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptorElamir = facerec.compute_face_descriptor(img, shape)


print(face_descriptorElamir)






img = io.imread('daryn.jpg')


win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptorDaryn = facerec.compute_face_descriptor(img, shape)

print(face_descriptorDaryn)





img = io.imread('rosa.jpg')


win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptorRosa = facerec.compute_face_descriptor(img, shape)

print(face_descriptorRosa)



img = io.imread('enzhu.jpg')


win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptorEnzhu = facerec.compute_face_descriptor(img, shape)

print(face_descriptorEnzhu)




img = io.imread('dayana.jpg')


win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptorDayana = facerec.compute_face_descriptor(img, shape)

print(face_descriptorDayana)




while(True):
    for i in range(30):
        cap.read()

    ret, frame = cap.read()

    cv2.imwrite('cam.png', frame)


    img = io.imread('cam.png')
    win2 = dlib.image_window()
    win2.clear_overlay()
    win2.set_image(img)
    dets_webcam = detector(img, 1)
    for k, d in enumerate(dets_webcam):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        win2.clear_overlay()
        win2.add_overlay(d)
        win2.add_overlay(shape)


    face_descriptorCam = facerec.compute_face_descriptor(img, shape)



    a = distance.euclidean(face_descriptorElamir, face_descriptorCam)
    if(a > 0.48):
        b = distance.euclidean(face_descriptorDaryn, face_descriptorCam)
        if(b > 0.48):
            c = distance.euclidean(face_descriptorRosa, face_descriptorCam)
            if(c > 0.48):
                d = distance.euclidean(face_descriptorEnzhu, face_descriptorCam)
                if(d > 0.48):
                    e = distance.euclidean(face_descriptorDayana, face_descriptorCam)
                    if(e > 0.48):
                        print("Dayana")
                    else:
                        print("None")
                else:
                    print("Enzhu")
            else:
                print("Rosa")
        else:
            print("Daryn")
    else:
        print("Elamir")

        
    os.remove("cam.png")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

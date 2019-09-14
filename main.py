from __future__ import division
from tkinter import *
import cv2
import dlib
import time
import sys
import numpy as np

root = Tk()

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):

    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    bboxes = []
    for (i, faceRect) in enumerate(faceRects):

        cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),
                  int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)

        shape = predictor(gray, faceRect)
        shape = shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frameDlibHog, (int(x*scaleWidth), int(y*scaleHeight)), 1, (0, 0, 255), -1)

    return frameDlibHog, bboxes

def openWebCam():
    global predictor
    hogFaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 0 )
    vid_writer = cv2.VideoWriter('output-hog-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

    frame_count = 0
    tt_dlibHog = 0
    while(1):
        hasFrame, frame2 = cap.read()
        frame = cv2.flip( frame2, 1 )
        if not hasFrame:
            break
        frame_count += 1

        t = time.time()
        outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
        tt_dlibHog += time.time() - t
        fpsDlibHog = frame_count / tt_dlibHog

        cv2.imshow("EmotiCapture", outDlibHog)

        vid_writer.write(outDlibHog)
        if frame_count == 1:
            tt_dlibHog = 0

        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    vid_writer.release()

root.geometry('%sx%s' % (1200, 1000))
root.configure(background="azure")

title = Label(root, text="EmotiCapture :)", font="System 40", background="azure")
title.pack(pady=50)
webCamButton = Button(root, text="Open Webcam", font="System 20", background="red2", command=openWebCam)
webCamButton.pack()
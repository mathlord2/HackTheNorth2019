<<<<<<< HEAD
from __future__ import division
from tkinter import *
import cv2
import dlib
import time
import sys
import imutils
import numpy as np
from PIL import Image, ImageTk
from vision import getImg
import pyperclip

#Set initial values
root = Tk()
emojiExists = False
angryFace = ImageTk.PhotoImage(Image.open("angryFace.png"))
contemptFace = ImageTk.PhotoImage(Image.open("contemptFace.png"))
disgustedFace = ImageTk.PhotoImage(Image.open("disgustedFace.png"))
fearFace = ImageTk.PhotoImage(Image.open("fearFace.png"))
happyFace = ImageTk.PhotoImage(Image.open("happyFace.png"))
neutralFace = ImageTk.PhotoImage(Image.open("neutralFace.png"))
sadFace = ImageTk.PhotoImage(Image.open("sadFace.png"))
surprisedFace = ImageTk.PhotoImage(Image.open("surprisedFace.png"))

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
        #for (x, y) in shape:
            #cv2.circle(frameDlibHog, (int(x*scaleWidth), int(y*scaleHeight)), 1, (255, 255, 255), -1)

    return frameDlibHog, bboxes

def openWebCam():
    global predictor, cap, frame_count, hogFaceDetector, tt_dlibHog, closeButton

    hogFaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 375)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 0 )

    frame_count = 0
    tt_dlibHog = 0

    closeButton = Button(root, text="Pause Webcam", font="System 15", background="red2", command=closeWebCam) #CLOSE BUTTON
    closeButton.pack(pady=5)

    showFrame()

def showFrame():
    global cap, hasFrame, frame2, frame_count, hogFaceDetector, tt_dlibHog, mainlabel, imgtk, outDlibHog, emojiTitle, emojiText, emojiExists, emoji
    mainlabel = Label(root)
    mainlabel.place(relx=0.5, rely=0.65, anchor='center')

    if emojiExists:
        emojiTitle.destroy()
        emojiText.destroy()
        emoji.destroy()
    
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 1 )

    frame_count += 1
        
    t = time.time()
    outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
    tt_dlibHog += time.time() - t
    fpsDlibHog = frame_count / tt_dlibHog


    cv2image = cv2.cvtColor(outDlibHog, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    mainlabel.imgtk = imgtk
    mainlabel.configure(image=imgtk)

    if frame_count == 1:
        tt_dlibHog = 0

    mainlabel.after(10, showFrame)
    

def closeWebCam(): #SOMEONE PROGRAM A CLOSE BUTTON
    global imageLink, closeButton, mainlabel, imgtk, outDlibHog, emojiTitle, emojiText, emojiExists, emoji

    mainlabel.destroy()
    closeButton.destroy()
    cv2.destroyAllWindows()

    cv2.imwrite("images/" + "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(outDlibHog, cv2.COLOR_BGR2RGBA))

    emotions = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    emojis = [angryFace, contemptFace, disgustedFace, fearFace, happyFace, neutralFace, sadFace, surprisedFace]
    codes = ["\U0001F620", "\U0001F612", "\U0001F922", "\U0001F628", "\U0001F603", "\U0001F610", "\U0001F622", "\U0001F631"]
    probabilities = list(getImg("images/" + "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg")[0]['faceAttributes']['emotion'].values())
    maxIndex = probabilities.index(max(probabilities))

    emojiTitle = Label(root, text="Your Emoji:", font="System 20", background="white")
    emojiTitle.pack(pady=2)

    emojiText = Label(root, text=(emotions[maxIndex] + " (" + str(probabilities[maxIndex]*100) + "% accuracy)"), font="System", background="white")
    emojiText.place(x=450, y=235)

    emoji = Label(image=emojis[maxIndex]) #Actual emoji
    emoji.place(x=680, y=230)

    pyperclip.copy(codes[maxIndex]) #Copy to clipboard

    emojiExists = True #Deletes emoji information next time webcam is run
    
root.geometry('%sx%s' % (1200, 1000))
root.configure(background="white")

image = Image.open("Easymoji_logo.png")
logo = ImageTk.PhotoImage(image)
logoLabel = Label(image=logo)
logoLabel.pack(pady=20)

webCamButton = Button(root, text="Open Webcam", font="System 15", background="red2", command=openWebCam)
webCamButton.pack()

root.mainloop()
=======
from __future__ import division
from tkinter import *
import cv2
import dlib
import time
import sys
import imutils
import numpy as np
from PIL import Image, ImageTk
from vision import getImg
import pyperclip

#Set initial values
root = Tk()
emojiExists = False
angryFace = ImageTk.PhotoImage(Image.open("angryFace.png"))
contemptFace = ImageTk.PhotoImage(Image.open("contemptFace.png"))
disgustedFace = ImageTk.PhotoImage(Image.open("disgustedFace.png"))
fearFace = ImageTk.PhotoImage(Image.open("fearFace.png"))
happyFace = ImageTk.PhotoImage(Image.open("happyFace.png"))
neutralFace = ImageTk.PhotoImage(Image.open("neutralFace.png"))
sadFace = ImageTk.PhotoImage(Image.open("sadFace.png"))
surprisedFace = ImageTk.PhotoImage(Image.open("surprisedFace.png"))

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
        #for (x, y) in shape:
            #cv2.circle(frameDlibHog, (int(x*scaleWidth), int(y*scaleHeight)), 1, (255, 255, 255), -1)

    return frameDlibHog, bboxes

def openWebCam():
    global predictor, cap, frame_count, hogFaceDetector, tt_dlibHog, closeButton

    hogFaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 375)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 0 )

    frame_count = 0
    tt_dlibHog = 0

    closeButton = Button(root, text="Pause Webcam", font="System 15", background="red2", command=closeWebCam) #CLOSE BUTTON
    closeButton.pack(pady=5)

    showFrame()

def showFrame():
    global cap, hasFrame, frame2, frame_count, hogFaceDetector, tt_dlibHog, mainlabel, imgtk, outDlibHog, emojiTitle, emojiText, emojiExists, emoji
    mainlabel = Label(root)
    mainlabel.place(relx=0.5, rely=0.65, anchor='center')

    if emojiExists:
        emojiTitle.destroy()
        emojiText.destroy()
        emoji.destroy()
    
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 1 )

    frame_count += 1
        
    t = time.time()
    outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector,frame)
    tt_dlibHog += time.time() - t
    fpsDlibHog = frame_count / tt_dlibHog


    cv2image = cv2.cvtColor(outDlibHog, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    mainlabel.imgtk = imgtk
    mainlabel.configure(image=imgtk)

    if frame_count == 1:
        tt_dlibHog = 0

    mainlabel.after(10, showFrame)
    

def closeWebCam(): #SOMEONE PROGRAM A CLOSE BUTTON
    global imageLink, closeButton, mainlabel, imgtk, outDlibHog, emojiTitle, emojiText, emojiExists, emoji

    mainlabel.destroy()
    closeButton.destroy()
    cv2.destroyAllWindows()

    cv2.imwrite("images/" + "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(outDlibHog, cv2.COLOR_BGR2RGBA))

    emotions = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    emojis = [angryFace, contemptFace, disgustedFace, fearFace, happyFace, neutralFace, sadFace, surprisedFace]
    codes = ["\U0001F620", "\U0001F612", "\U0001F922", "\U0001F628", "\U0001F603", "\U0001F610", "\U0001F622", "\U0001F631"]
    probabilities = list(getImg("images/" + "frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg")[0]['faceAttributes']['emotion'].values())
    maxIndex = probabilities.index(max(probabilities))

    emojiTitle = Label(root, text="Your Emoji:", font="System 20", background="white")
    emojiTitle.pack(pady=2)

    emojiText = Label(root, text=(emotions[maxIndex] + " (" + str(probabilities[maxIndex]*100) + "% accuracy)"), font="System", background="white")
    emojiText.place(x=450, y=235)

    emoji = Label(image=emojis[maxIndex]) #Actual emoji
    emoji.place(x=680, y=230)

    pyperclip.copy(codes[maxIndex]) #Copy to clipboard

    emojiExists = True #Deletes emoji information next time webcam is run
    
root.geometry('%sx%s' % (1200, 1000))
root.configure(background="white")

image = Image.open("Easymoji_logo.png")
logo = ImageTk.PhotoImage(image)
logoLabel = Label(image=logo)
logoLabel.pack(pady=20)

webCamButton = Button(root, text="Open Webcam", font="System 15", background="red2", command=openWebCam)
webCamButton.pack()

root.mainloop()
>>>>>>> origin

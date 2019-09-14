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
            cv2.circle(frameDlibHog, (int(x*scaleWidth), int(y*scaleHeight)), 1, (255, 255, 255), -1)

    return frameDlibHog, bboxes

def openWebCam():
    global predictor, cap, frame_count, hogFaceDetector, tt_dlibHog, closeButton

    hogFaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    hasFrame, frame2 = cap.read()
    frame = cv2.flip( frame2, 0 )

    frame_count = 0
    tt_dlibHog = 0

    closeButton = Button(root, text="Pause Webcam", font="System 15", background="red2", command=closeWebCam) #CLOSE BUTTON
    closeButton.pack(pady=5)

    showFrame()

def showFrame():
    global cap, hasFrame, frame2, frame_count, hogFaceDetector, tt_dlibHog, mainlabel, imgtk
    mainlabel = Label(root)
    mainlabel.place(relx=0.5, rely=0.6, anchor='center')
    
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
    global imageLink, closeButton, mainlabel, imgtk
    mainlabel.destroy()
    closeButton.destroy()
    cv2.destroyAllWindows()
    if imgtk.mode != "RGBA":
        imgtk = imgtk.convert("RGBA")
    txt = Image.new('RGBA', imgtk.size, (255,255,255,0))

    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype("arial.ttf", fontsize)
    draw.text((0, 0),caption,(255,0,0),font=font)

    file = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
    if file:
        abs_path = os.path.abspath(file.name)
        out = Image.alpha_composite(imgtk, txt)
        out.save(abs_path) # saves the image to the input file name

    emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    emojis = ["U0001F621", "U0001F614", "U0001F616", "U0001F631", "U0001F604", "U0001F610", "U0001F622", "U0001F632"]
    probabilities = list(getImg(file.name)[0]['faceAttributes']['emotion'].values())
    maxIndex = probabilities.index(max(probabilities))

    emojiTitle = Label(root, text="Your Emoji:", font="System 20", background="azure")
    emojiTitle.pack()

    emojiText = Label(root, text=(emotions[maxIndex]+' ' + emojis[maxIndex]))
    emojiText.pack()

root.geometry('%sx%s' % (1200, 1000))
root.configure(background="azure")

image = Image.open("emoticapture.png")
logo = ImageTk.PhotoImage(image)
logoLabel = Label(image=logo)
logoLabel.pack(pady=20)

webCamButton = Button(root, text="Open Webcam", font="System 15", background="red2", command=openWebCam)
webCamButton.pack()

root.mainloop()

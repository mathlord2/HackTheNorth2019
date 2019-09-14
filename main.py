from tkinter import *
from random import randint
root = Tk()
screen = Canvas(root, width=48, height=48, background="white")
screen.pack()
import pixelProcess as pp

# To use pixelProcess, use the pp.process() function and give the tkinter canvas and pixel list parameters.

dataset = pp.getDataSet("fer2013.csv")
dots = dataset[randint(1, 10000)][1]

pp.process(screen,dots)
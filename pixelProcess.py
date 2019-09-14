#A functional file to process csv pixel information

def rgbhex(num):
    hex = "#%02x%02x%02x" % (num, num, num)
    return hex

def process(screen, dots):
    count = 0
    for i in range(48):
        for j in range(48):
            screen.create_oval(j,i,j,i,fill=rgbhex(dots[count]), outline=rgbhex(dots[count]), width=1)
            count += 1

    screen.update()

    while True:
        pass
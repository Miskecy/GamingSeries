import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture
from vision import Vision

# initialize the WindowCapture
wincap = WindowCapture('Legends Of Idleon')

# load the trained model
cascade_melty = cv.CascadeClassifier('src/cascade/cascade.xml')

# load an empty Vision Class
vision_melty = Vision(None)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    if screenshot is None:
        continue

    # do object detection
    rectangles = cascade_melty.detectMultiScale(screenshot)

    # draw the detection results onto the original image
    detection_image = vision_melty.draw_rectangles(screenshot, rectangles)

    # display the processed image
    cv.imshow('Legends of the BotOn', screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image.
    # press 'd' to save as a negative image.
    # waits 1 ms every loop to process key press
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('img/positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('d'):
        cv.imwrite('img/negative/{}.jpg'.format(loop_time), screenshot)


print('Done.')

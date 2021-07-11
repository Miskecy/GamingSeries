import cv2 as cv
import numpy as np
from time import time
from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter

# initialize the WindowCapture
wincap = WindowCapture('Legends Of Idleon')
# initialize the Vision class
vision_melty = Vision('img/idleon_melty_cube_hsv_filter.jpg')
# initialize the trackbar window
vision_melty.init_control_gui()

# melty cube HSV filter
hsv_filter = HsvFilter(70, 152, 114, 104, 255, 255, 255, 91, 34, 63)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    if screenshot is None:
        continue

    # pre-process the image
    processed_image = vision_melty.apply_hsv_filter(screenshot, hsv_filter)

    # do object detection
    rectangles = vision_melty.find(processed_image, 0.5)

    # draw the detection results into the original image
    output_image = vision_melty.draw_rectangles(processed_image, rectangles)

    # display the processed image
    cv.imshow('Matches', output_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key press
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break


print('Done.')

import cv2 as cv
from time import time
from windowcapture import WindowCapture

wincap = WindowCapture('Legends Of Idleon')

loop_time = time()
while(True):

    screenshot = wincap.get_screenshot()

    cv.imshow('Computer Vision', screenshot)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key press
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break


print('Done.')

import cv2 as cv
import numpy as np
from PIL import ImageGrab
from time import time

loop_time = time()
while(True):

    screenshot = ImageGrab.grab()
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    cv.imshow('Computer Vision', screenshot)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key press
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')

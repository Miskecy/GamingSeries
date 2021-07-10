import cv2 as cv
import numpy as np
# import os

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import sys
# np.set_printoptions(threshold=sys.maxsize)

haystack_img = cv.imread('src/idleon_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('src/idleon_cooler.jpg', cv.IMREAD_UNCHANGED)

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
print(result)

threshold = 0.40
locations = np.where(result >= threshold)
# the np.where() return value will look like this:
# (array([226, 286], dtype=int64), array([ 98, 422], dtype=int64))
print(locations)

# we can zip those up into position tuples
locations = list(zip(*locations[::-1]))
# zip explanation:
# res = [[10, 20, 30], [7, 8, 9]]
# res[::-1]
# [[7, 8, 9], [10, 20, 30]]
# zip(*res[::-1])
# <zip object at 0x012DEB28>
# list(zip(*res[::-1]))
# [(7, 10), (8, 20), (9, 30)]
# print(locations)

# first we need to create the list of [x, y, w, h] rectangles
rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)
    rectangles.append(rect)

rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
print(rectangles)

if len(rectangles):
    print('Found needle.')

    line_color = (0, 255, 0)
    line_thickness = 2
    line_type = cv.LINE_4
    marker_color = (255, 0, 255)
    marker_thickness = 2
    marker_type = cv.MARKER_CROSS

    # need to loop over all the locations and draw their rectangle
    for (x, y, w, h) in rectangles:
        '''
        # determine the box positions
        top_left = (x, y)
        bottom_right = (x + w, y + h)

        # draw the box
        cv.rectangle(haystack_img, top_left, bottom_right,
                     line_color, line_thickness, line_type)
        '''
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        cv.drawMarker(haystack_img, (center_x, center_y),
                      marker_color, marker_type)
    # cv.imwrite('result.jpg', haystack_img)
    cv.imshow('Matches', haystack_img)
    cv.waitKey()

else:
    print('Needles not found.')

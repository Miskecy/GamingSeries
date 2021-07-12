import os


def generate_negative_description_file():
    # open the output file for writing. Will overwrite all existing data in there
    with open('src/neg.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir('img/negative'):
            f.write('img/negative/' + filename + '\n')


generate_negative_description_file()


# the opencv_annotation executable can befound in opencv/build/x64/vc15/bin
# generate positive description file using:
# F:\Path\opencv_annotation.exe --annotations=src/pos.txt --images=img/positive/

# mark rectangles with the left mouse button,
# press 'c' to accept a selection,
# press 'd' to delete the latest selection,
# press 'n' to proceed with next image,
# press 'esc' to stop.

# Create Samples
# F:\Path\opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

# Create Cascade
# F:\Path\opencv_traincascade.exe -data src/cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -numPos 200 -numNeg 100 -numStages 10

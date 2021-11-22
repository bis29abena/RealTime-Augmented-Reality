# Usage
# python main.py --video "video file"

# import the necessary packages
from bismar_AR.AR import find_and_warp
from imutils.video import VideoStream
from collections import deque
import cv2 as cv
import imutils
import argparse
import time

# construct an argparse to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, type=str, help="Path to video file for AR")
ap.add_argument("-c", "--cache", default=-1, type=int, help="whether or not to use reference point cache")
args = vars(ap.parse_args())

# load the aruco dictionary and parse the Aruco parameters
print("[INFO] initialising marker detector")
arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_50)
arucoParams = cv.aruco.DetectorParameters_create()

# initialize the video file stream
print("[INFO] accessing video stream")
vf = cv.VideoCapture(args["video"])

# initialise a queue to maintain the next frame from the video stream
Q = deque(maxlen=128)

# we need a from our queue to start our augmented reality pipeline
# so read the next frame from our video file source and ad it to our queue
(grabbed, source) = vf.read()
Q.appendleft(source)

# we initialise our video stream and allow the camera sensor to warm up
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while len(Q) > 0:
    # grab the frame from our video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    # attempt to find the Aruco markers in the frame, and provided
    # they are found, take the current source image and warp it unto
    # input frame using our augmented reality technique.
    warped = find_and_warp(frame=frame, source=source, cornersID=(1, 2, 4, 3),
                           arucoDict=arucoDict, arucoParams=arucoParams,
                           useCache=args["cache"] > 0)

    # if the warped frame is not None, then we know (1) we found the
    # four Aruco markers and (2) the perspective warped was finally applied
    if warped is not None:
        # set the frame to the output AR frame and then
        # grab the next video file frame from our queue
        frame = warped
        source = Q.popleft()

    # for speed/efficiency, we can use a queue to keep the next video frame
    # ready for us.
    if len(Q) != Q.maxlen:
        # read the next frame from the video file stream
        (grabbed, nextFrame) = vf.read()

        # if the frame was read add the frame to our queue
        if grabbed:
            Q.appendleft(nextFrame)

    # show the output frame
    cv.imshow("frame", frame)
    key = cv.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

cv.destroyAllWindows()
vs.stop()

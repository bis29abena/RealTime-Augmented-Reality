# import necessary packages
import numpy as np
import cv2 as cv

# initialised cached reference point
CACHED_REF_PTS = None


def find_and_warp(frame, source, cornersID, arucoDict, arucoParams, useCache=False):
    # grab a reference to our cache reference
    global CACHED_REF_PTS

    # grab the width and height of the source frame respectively
    (FH, FW) = frame.shape[:2]
    (SH, SW) = source.shape[:2]

    # Detect Aruco markers in the input frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # if we did not find the 4 corners initialise an empty ids list, otherwise flatted the ids
    ids = np.array([]) if len(corners) != 4 else ids.flatten()

    # initialise our list of reference points
    refPts = []

    # loop over the corner ids of the Aruco Markers in tl, tr, br, bl order
    for i in cornersID:
        # grab the index of the corner with the current id
        j = np.squeeze(np.where(ids == i))

        # id we receive an empty list instead of a integer index
        # the we could not find the ids
        if j.size == 0:
            continue

        # otherwise append the corner append the corner x, y co-ordinates to our list of refPTS
        corner = np.squeeze(corners[j])
        refPts.append(corner)

    # check to see if we fail to find all the four Aruco markers
    if len(refPts) != 4:
        # if we allowed to use the cache refPts fall back on them
        if useCache and CACHED_REF_PTS is not None:
            refPts = CACHED_REF_PTS

        # otherwise we cannot use the cache or there are no previous
        # cached refPts, return early
        else:
            return None

    # if we are allowed to use the cached reference points, then update
    # the update the cache with the current set
    if useCache:
        CACHED_REF_PTS = refPts

    # unpack our Aruco reference points and use the the reference the points to define the desitination transform
    # matrix, making sure that the points are specified in tl, tr, br, bl order
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)


    # Grab the spatial dimensions of the source mage and define the transform matrix for the source image
    # in tl, tr, br, bl order
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]])

    # Compute the homography matrix and then warp the source image to the destination based on the homography
    (H, _) = cv.findHomography(srcMat, dstMat)
    warped = cv.warpPerspective(source, H, (FW, FH))


    # construct a mask of the source image now that the perspective warp has taken place
    # (we'll need this mask to copy the source image into the destination)
    mask = np.zeros((FH, FW), dtype="uint8")
    cv.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv.LINE_AA)

    # Create a 3 channel version of the mask by stacking it depth-wise,
    # such that we can copy the warped source image into the input
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)

    # copy the wraped source image into the input image by 1) multiplying the masked image and masked together,
    # 2) multiplying the the original input with the maskand 3) adding the results multiplication together.
    warpedMultiplied = cv.multiply(warped.astype(float), maskScaled)
    image_multiply = cv.multiply(frame.astype(float), 1.0 - maskScaled)
    output = cv.add(warpedMultiplied, image_multiply)
    output = output.astype("uint8")
    return output

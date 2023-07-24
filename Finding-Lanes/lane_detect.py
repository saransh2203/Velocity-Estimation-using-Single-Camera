import cv2
import numpy as np
import utils

def getLaneCurve(img):
    imgCopy = img.copy()
    # STEP1
    imgThres = utils.thresholding(img)

    # STEP2
    h, w, c = img.shape
    points = utils.valTrackbars() 
    imgWarp = utils.warpImg(imgThres, points, w, h)
    imgWarpPoints = utils.drawPoints(img, points)

    #STEP3
    basePoint, imgHist = utils.getHistogram(imgWarp, display=True)

    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Warp Points', imgWarpPoints)
    cv2.imshow('Histogram', imgHist)

    # Obtain photonegative of the threshold image
    imgThresNegative = cv2.bitwise_not(imgThres)
    cv2.imshow('Thres Negative', imgThresNegative)

    # Obtain warped image of the photonegative
    imgWarpNegative = utils.warpImg(imgThresNegative, points, w, h)
    cv2.imshow('Warp Negative', imgWarpNegative)

    return None

if __name__ == '__main__':
    cap = cv2.VideoCapture('test-video-road.mp4')
    initialTrackBarVals = [100, 100, 100, 100]
    utils.initializeTrackbars(initialTrackBarVals)

    while True:
        success, img = cap.read()
        if not success:
            # Reached the end of the video, restart from the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        img = cv2.resize(img, (480, 240))
        getLaneCurve(img)

        cv2.imshow('Vid', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

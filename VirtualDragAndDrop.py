import math
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()

    # detect hands
    hands = detector.findHands(img, draw=False)

    if len(hands) > 0:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmarks points
        # bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPoint1 = hand1["center"]  # center of the hand cx,cy
        # handType1 = hand1["type"]  # Hand Type Left or Right
        fingers1 = detector.fingersUp(hand1)

        if sum(fingers1) == 2 and fingers1[1] == 1 and fingers1[2] == 1:
            x1, y1 = lmList1[8][0], lmList1[8][1]
            x2, y2 = lmList1[12][0], lmList1[12][1]
            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            print(length)

            if length < 100:
                cursor = lmList1[8]  # index finger tip landmark
                cv2.circle(img, (cursor[0], cursor[1]), 30, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

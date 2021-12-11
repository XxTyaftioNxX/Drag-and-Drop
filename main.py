from typing import NewType
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,  720)

detector = HandDetector(detectionCon=0.8, maxHands=2)

class DragObject():
    def __init__(self, posCenter, size=[150, 150], colorR=(255, 0, 255)):
        self.posCenter = posCenter
        self.size = size
        self.colorR = colorR

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        if cx-w//2 < cursor[0] < cx+w//2 and cy-h//2 < cursor[1] < cy+h//2:
            self.posCenter = cursor
            self.colorR = (0, 0, 0)
        
        else:
            self.colorR = (255, 0, 255)

#creating several instances 
rectList = []
for x in range(5):
    rectList.append(DragObject([x*200+150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img,flipType=False)

    #hands = detector.findHands(img, flipType=False)

    if hands:

        l, _, _ = detector.findDistance(hands[0]['lmList'][8], hands[0]['lmList'][12], img)

        if l < 30:

            cursor = hands[0]['lmList'][8]
            #calling object Function
            for rect in rectList:
                rect.update(cursor)

    '''
    #draw rectangle (solid)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size 

        cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), rect.colorR, cv2.FILLED)

        cvzone.cornerRect(img, (cx-w//2, cy-h//2, w, h), 20, rt=0)
    '''

    #draw rectangle transparent
    imgNew = np.zeros_like(img, np.uint8)

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size 

        cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), rect.colorR, cv2.FILLED)

        cvzone.cornerRect(imgNew, (cx-w//2, cy-h//2, w, h), 20, rt=0)

    out = img.copy()
    alpha=0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)

    #quit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
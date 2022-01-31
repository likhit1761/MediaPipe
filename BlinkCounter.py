import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(frame, face[id], 2, color, cv2.FILLED)
        leftup = face[159]
        leftdown = face[23]
        leftleft = face[130]
        leftright = face[243]

        lengthVer, _ = detector.findDistance(leftup, leftdown)
        lengthHor, _ = detector.findDistance(leftleft, leftright)

        cv2.line(frame, leftup, leftdown, (0, 200, 0), 2)
        cv2.line(frame, leftleft, leftright, (0, 200, 0), 2)
        ratio = int(100 * (lengthVer / lengthHor))
        ratioList.append(ratio)

        if (len(ratioList) > 5):
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            counter = 1
            color = (0, 200, 0)
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (255, 0, 255)

        cvzone.putTextRect(frame, f'Blink Count : {blinkCounter}', (50, 100), colorR=color)

        # framePlot = plotY.update(ratio)
        framePlot = plotY.update(ratioAvg, color)
        # cv2.imshow("PLot", framePlot)
        imgStack = cvzone.stackImages([frame, framePlot], 2, 1)
    else:
        imgStack = cvzone.stackImages([frame, frame], 2, 1)

    cv2.imshow("Stack", imgStack)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

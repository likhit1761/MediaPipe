import cvzone
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
while True:
    _, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        pointleft = face[145]
        pointright = face[374]
        cv2.circle(img, pointleft, 3, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, pointright, 3, (255, 0, 255), cv2.FILLED)
        cv2.line(img, pointleft, pointright, (255, 0, 255), 1)
        wsmall, _ = detector.findDistance(pointleft, pointright)
        W = 6.3
        f = 840
        d = (W * f) / wsmall
        print(d)
        cvzone.putTextRect(img, f"{int(d)}cm", (face[10][0]-50, face[10][1]-10), scale=2)
    # img = cv2.flip(img, 1)
    cv2.imshow("frame", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

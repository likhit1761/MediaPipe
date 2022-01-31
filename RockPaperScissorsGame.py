import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def getHandMove(land_handmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i - 3].y for i in range(9, 20, 4)]):
        return "rock"
    elif landmarks[13].y < landmarks[17].y and landmarks[17].y < landmarks[20].y:
        return "scissors"
    else:
        return "paper"


vid = cv2.VideoCapture(0)
clock = 0
p1_move = p2_move = None
gametext = ""
success = True

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ok, frame = vid.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        frame = cv2.flip(frame, 1)
        if 0 <= clock < 20:
            success = True
            gametext = "ready"
        elif clock < 30:
            gametext = "3...."
        elif clock < 40:
            gametext = "2..."
        elif clock < 50:
            gametext = "1..."
        elif clock < 60:
            gametext = "Go..."
        elif clock == 60:
            hls = results.multi_hand_landmarks
            if hls and len(hls) == 2:
                p1_move = getHandMove(hls[0])
                p2_move = getHandMove(hls[1])
            else:
                success = False
        elif clock < 100:
            if success:
                gametext = f"p1 is {p1_move}, p2 is {p2_move}"
                if p1_move == p2_move:
                    gametext = "Tied"
                elif p1_move == "paper" and p2_move == "rock":
                    gametext = "player 1 win"
                elif p1_move == "rock" and p2_move == "scissors":
                    gametext = "player 1 win"
                elif p1_move == "scissors" and p2_move == "paper":
                    gametext = "player 1 win"
                else:
                    gametext = "player 2 win"
            else:
                gametext = "Not proper"
        cv2.putText(frame, f"clock: {clock}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, gametext, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        clock = (clock + 1) % 100
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()

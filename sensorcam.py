import cv2
import mediapipe as mp



cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    results_hands = hands.process(image)
  
    results_face = face_mesh.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  
    def is_finger_up(landmarks, finger_tip, finger_dip):
        return landmarks[finger_tip].y < landmarks[finger_dip].y

    
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    finger_dips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_DIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp_hands.HandLandmark.RING_FINGER_DIP,
        mp_hands.HandLandmark.PINKY_DIP
    ]

    finger_names = ['Jempol', 'Telunjuk', 'Tengah', 'Manis', 'Kelingking']

   
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for i, (tip, dip) in enumerate(zip(finger_tips, finger_dips)):
                if is_finger_up(hand_landmarks.landmark, tip, dip):
                    print(f"{finger_names[i]} diangkat")
                    

    
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            
            left_eye = [face_landmarks.landmark[i] for i in range(133, 144)]
            right_eye = [face_landmarks.landmark[i] for i in range(362, 373)]
            if sum([p.y for p in left_eye]) / len(left_eye) < 0.3:  
                print("Mata kiri dikedipkan")
                
            if sum([p.y for p in right_eye]) / len(right_eye) < 0.3:  
                print("Mata kanan dikedipkan")
                

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



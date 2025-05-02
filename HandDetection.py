import cv2
import mediapipe as mp

# Initialiser les modules MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Paramètres de détection
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2  # Nombre maximum de mains à détecter
)

# Initialiser la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convertir l'image en RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Détection des mains
    results = hands.process(image)

    # Reconvertir en BGR pour l'affichage
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dessiner les résultats
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

    # Afficher l'image
    cv2.imshow('Hand Detection', image)
    
    # Quitter avec la touche 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libérer les ressources
hands.close()
cap.release()
cv2.destroyAllWindows()
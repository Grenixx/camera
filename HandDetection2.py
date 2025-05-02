import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Chemin vers le bundle .task
MODEL_PATH = "gesture_recognizer.task"

# 2. Configuration de BaseOptions
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

# 3. Options pour le GestureRecognizer en mode vidéo
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode       = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO     # mode flux vidéo :contentReference[oaicite:0]{index=0}
)

# 4. Création de l’objet recognizer
with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)  # 0 pour la webcam
    if not cap.isOpened():
        raise RuntimeError("Impossible d’ouvrir la webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Conversion en mp.Image pour MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        # Inference bloquante pour la vidéo
        result = recognizer.recognize_for_video(
            mp_image,
            timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Extraction du geste le plus probable
        if result.gestures and result.gestures[0]:
            gesture = result.gestures[0][0]
            label   = gesture.category_name
            score   = gesture.score
        else:
            label, score = "aucun", 0.0

        # Affichage du label et confiance
        cv2.putText(
            frame,
            f"{label} ({score:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

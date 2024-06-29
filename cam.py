import cv2
import mediapipe as mp
import numpy as np

# Inicializar utilitários e soluções do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializar o VideoCapture com a URL da câmera IP
ip = "http://192.168.18.2:4747/video"
cap = cv2.VideoCapture(ip)

# Verifique se a captura de vídeo foi inicializada corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Configurar o detector de mãos do MediaPipe
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    # Criação de uma tela para desenhar
    drawing = None
    drawing_color = (0, 0, 255)  # Vermelho
    drawing_thickness = 10  # Aumentar a espessura

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando quadro vazio da câmera.")
            continue

        # Converter a imagem para RGB (MediaPipe usa RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processar a detecção de mãos
        results = hands.process(image)

        # Converter de volta para BGR para exibir corretamente com OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Inicializar a tela de desenho
        if drawing is None:
            drawing = np.zeros_like(image)

        # Desenhar as anotações de mãos na imagem
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar as landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Pegar a coordenada do ponto desejado (ex: ponta do dedo indicador)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                h, w, _ = image.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Verificar se o dedo indicador está acima dos outros dedos
                if (index_finger_tip.y < middle_finger_tip.y and
                        index_finger_tip.y < ring_finger_tip.y and
                        index_finger_tip.y < pinky_finger_tip.y and
                        index_finger_tip.y < thumb_tip.y):
                    # Desenhar na tela de desenho apenas se o dedo indicador estiver acima dos outros dedos
                    cv2.circle(drawing, (x, y), drawing_thickness, drawing_color, -1)

        # Combinar a tela de desenho com a imagem original
        combined_image = cv2.addWeighted(image, 0.5, drawing, 0.5, 0)

        # Mostrar a imagem com as anotações e desenhos
        cv2.imshow('MediaPipe Hands Drawing', combined_image)

        # Pressione 'Esc' para sair do loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Liberar a captura de vídeo e destruir todas as janelas
cap.release()
cv2.destroyAllWindows()

import mediapipe as mp
import cv2
import numpy as np

class HeadTracker:
    def __init__(self):
        # Inicialización de MediaPipe para la detección facial
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, 
                                                     max_num_faces=1,
                                                     refine_landmarks=True,
                                                     min_detection_confidence=0.5, 
                                                     min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    def get_eye_left_distance(self, frame):
        """
        Detecta la distancia del ojo izquierdo en un frame dado usando MediaPipe Face Mesh.
        Devuelve la distancia entre los puntos superior e inferior del ojo izquierdo.
        """
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Realizar la detección de la cara
        results = self.face_mesh.process(frame_rgb)
        #OK
        if results.multi_face_landmarks:
            # Obtener los puntos del ojo izquierdo en Face Mesh
            left_eye_top = results.multi_face_landmarks[0].landmark[133]  # Punto superior del ojo izquierdo
            left_eye_bottom = results.multi_face_landmarks[0].landmark[144]  # Punto inferior del ojo izquierdo
            # Obtener dimensiones de la imagen
            height, width, _ = frame.shape

            # Convertir coordenadas normalizadas a valores de píxeles
            left_eye_top_y = int(left_eye_top.y * height)
            left_eye_bottom_y = int(left_eye_bottom.y * height)

            # Calcular la distancia entre el ojo superior e inferior
            eye_distance = abs(left_eye_bottom_y - left_eye_top_y)

            return eye_distance

        return None

    def is_left_eye_winking(self, frame, threshold=1):
        """
        Detecta si el ojo izquierdo está guiñando comparando la distancia entre el ojo superior e inferior.
        Si la distancia es menor que un umbral, se considera que el ojo está cerrado (guiñado).
        """
        eye_distance = self.get_eye_left_distance(frame)
        
        if eye_distance is not None and eye_distance <= threshold:
            print(eye_distance, threshold)
            return True  # Ojo izquierdo guiñado
        print(False)
        return False  # Ojo izquierdo no guiñado
    def get_nose_position(self, frame):
        """
        Detecta la posición de la nariz en un frame dado usando MediaPipe Face Mesh.
        Devuelve las coordenadas (x, y) de la nariz si se detecta un rostro, de lo contrario, None.
        """
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Realizar la detección de la cara
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Obtener el punto 1, que corresponde a la nariz en Face Mesh
            nose_landmark = results.multi_face_landmarks[0].landmark[1]
            
            # Obtener dimensiones de la imagen
            height, width, _ = frame.shape
            
            # Convertir coordenadas normalizadas a valores de píxeles
            nose_x = int(nose_landmark.x * width)
            nose_y = int(nose_landmark.y * height)

            return (nose_x, nose_y)

        return None

    def is_mouth_open(self, frame):
        """
        Detecta si la boca está abierta en un frame dado usando MediaPipe Face Mesh.
        Devuelve True si la boca está abierta, False si está cerrada, y None si no se detecta un rostro.
        """
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Realizar la detección de la cara
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            # Obtener los puntos de la boca superior e inferior en Face Mesh
            mouth_top = results.multi_face_landmarks[0].landmark[13]  # Punto 13 de la boca superior
            mouth_bottom = results.multi_face_landmarks[0].landmark[14]  # Punto 14 de la boca inferior

            # Obtener dimensiones de la imagen
            height, width, _ = frame.shape

            # Convertir coordenadas normalizadas a valores de píxeles
            mouth_top_y = int(mouth_top.y * height)
            mouth_bottom_y = int(mouth_bottom.y * height)

            # Calcular la distancia entre la boca superior e inferior
            mouth_distance = abs(mouth_bottom_y - mouth_top_y)

            # Umbral de apertura de la boca
            if mouth_distance > 15:  # Ajusta este valor según lo necesario
                return True  # Boca abierta
            else:
                return False  # Boca cerrada

        return None  # No se detectó rostro

# Ejemplo de uso
# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)  # Usar la cámara por defecto
#     head_tracker = HeadTracker()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detectar la posición de la nariz
#         nose_position = head_tracker.get_nose_position(frame)
#         if nose_position:
#             cv2.circle(frame, nose_position, 5, (0, 255, 0), -1)

#         # Detectar si la boca está abierta
#         if head_tracker.is_mouth_open(frame):
#             cv2.putText(frame, "Mouth: OPEN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, "Mouth: CLOSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Mostrar la imagen con las anotaciones
#         cv2.imshow("Head Tracker", frame)

#         # Salir si se presiona la tecla 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

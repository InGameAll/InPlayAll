import cv2
import mediapipe as mp
import numpy as np
from evdev import UInput, AbsInfo, ecodes as e
from pynput import keyboard
import time
import threading

# Definir capacidades del control virtual
capabilities = {
    e.EV_ABS: [
        (e.ABS_RX, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)),
        (e.ABS_RY, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0))
    ],
    e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_DPAD_UP, e.BTN_X, e.BTN_Y]  # Botones de acción
}

ui = UInput(capabilities, name="Control De La Cara")
scaled_x, scaled_y = 0, 0


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Índice del punto de la nariz
NOSE_TIP = 1  

# Inicializa la cámara web
cap = cv2.VideoCapture(0)

_ancho, _alto = 640, 480
square_size = 20
canvas_mesh = np.zeros((_alto, _ancho, 3), dtype=np.uint8)
canvas_square = np.zeros((_alto, _ancho, 3), dtype=np.uint8)
canvas_square_2 = np.zeros((_alto, _ancho, 3), dtype=np.uint8)
aligned_frame = np.zeros((_alto, _ancho, 3), dtype=np.uint8)

square_x = 640 - square_size // 2
square_y = 480 - square_size // 2

previous_nose_position = None
stable_frames = 0
STABILITY_THRESHOLD = 100
calibration_points = []
calibrating = True

# Parámetros de control
DEAD_ZONE = 0.002  # Umbral de zona muerta
SMOOTHING_FACTOR = 0.006  # Factor de suavizado
MOVEMENT_THRESHOLD = 0.05  # Rango de corrección ortogonal

center_reset_counter = 0
center_x, center_y = _ancho / 2, _alto / 2


# Mapeo de teclas del teclado a botones del joystick virtual
key_map = {
    'x': e.BTN_X,  # Mapeado a botón del control
    'y': e.BTN_Y,
    'a': e.BTN_A,
    'b': e.BTN_B
}

def on_press(key):
    """ Captura el evento de tecla presionada y lo envía al control virtual """
    try:
        if key.char in key_map:
            print(f"[INFO] Tecla {key.char} presionada -> {key_map[key.char]}")
            ui.write(e.EV_KEY, key_map[key.char], 1)
            ui.syn()
    except AttributeError:
        pass  # Ignora teclas especiales que no son caracteres

def on_release(key):
    """ Captura el evento de tecla liberada y lo envía al control virtual """
    try:
        if key.char in key_map:
            print(f"[INFO] Tecla {key.char} liberada -> {key_map[key.char]}")
            ui.write(e.EV_KEY, key_map[key.char], 0)
            ui.syn()
    except AttributeError:
        pass

def start_keyboard_listener():
    """ Inicia el listener de teclas """
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# Iniciar el listener de teclas en un hilo separado
keyboard_thread = threading.Thread(target=start_keyboard_listener)
keyboard_thread.daemon = True  # Permite que el hilo termine cuando el programa principal termine
keyboard_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        canvas_square[:] = 0

    # Procesar imagen
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(color_frame)
    canvas_mesh[:] = 0
    canvas_square_2[:] = 0
    if results.multi_face_landmarks:
        
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(canvas_mesh, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            nose = face_landmarks.landmark[NOSE_TIP]
            nose_position = np.array([nose.x, nose.y])

            if calibrating:
                calibration_points.append(nose_position)
                if len(calibration_points) > 50:
                    calibrating = False  # Termina la calibración después de 50 muestras
            else:
                if previous_nose_position is not None:
                    movement = nose_position - previous_nose_position

                    if abs(movement[0]) > DEAD_ZONE or abs(movement[1]) > DEAD_ZONE:
                        stable_frames = 0
                        
                        # Corrección ortogonal
                        if abs(movement[0]) > abs(movement[1]) and abs(movement[0]) > MOVEMENT_THRESHOLD:
                            if movement[0] > 0:
                                scaled_x, scaled_y = -int((nose_position[0] - 0.5) * 65535) * 2.25,0
                            else:
                                scaled_x, scaled_y = -int((nose_position[0] - 0.5) * 65535) * 2.25,0
                            
                        elif abs(movement[1]) > abs(movement[0]) and abs(movement[1]) > MOVEMENT_THRESHOLD:
                            if movement[1] > 0:
                                scaled_x, scaled_y = 0, int((nose_position[1] - 0.5) * 65535) * 2.25
                            else:
                                scaled_x, scaled_y = 0, int((nose_position[1] - 0.5) * 65535) * 2.25
                            
                        
                        
                        # Suavizado del movimiento
                        scaled_x = int(previous_nose_position[0] * SMOOTHING_FACTOR + scaled_x * (1 - SMOOTHING_FACTOR))
                        scaled_y = int(previous_nose_position[1] * SMOOTHING_FACTOR + scaled_y * (1 - SMOOTHING_FACTOR))
                        
                        # Limitar los valores al rango del joystick
                        scaled_x = np.clip(scaled_x, -32768, 32768)
                        scaled_y = np.clip(scaled_y, -32768, 32768)
                        
                        # Reset al centro si se mantiene estable
                        if stable_frames >= STABILITY_THRESHOLD:
                            if center_reset_counter == 0:  # Solo reseteamos una vez cuando la estabilidad se mantiene
                                scaled_x, scaled_y = 0, 0
                                center_reset_counter += 1  # Evita reseteos adicionales
                        else:
                            center_reset_counter = 0 
                    else:
                        stable_frames += 1
                        print("Estable " + str(stable_frames))
                        if stable_frames >= STABILITY_THRESHOLD:
                            print("Establility umbral menors")
                            # Solo reseteamos a (0, 0) si se mantiene estable por más de STABILITY_THRESHOLD frames
                            if abs(scaled_x) < MOVEMENT_THRESHOLD and abs(scaled_y) < MOVEMENT_THRESHOLD:
                                scaled_x, scaled_y = 0, 0  # Reset al centro solo cuando el movimiento es mínimo
                                previous_nose_position = nose_position    
                                

                        
                else:
                    previous_nose_position = nose_position
                    stable_frames = 0  # Reiniciar estabilidad al detectar un rostro
                    
                ui.write(e.EV_ABS, e.ABS_RX, scaled_x)
                ui.write(e.EV_ABS, e.ABS_RY, scaled_y)
                ui.syn()
                
                # Mover la imagen para que la nariz esté en el centro
                offset_x = (nose_position[0] - 0.5) * _ancho  # Desplazamiento en X de la nariz respecto al centro
                offset_y = (nose_position[1] - 0.5) * _alto   # Desplazamiento en Y de la nariz respecto al centro
                
                # Trasladamos la imagen
                M = np.float32([[1, 0, offset_x], [0, 1, -offset_y]])  # Matriz de transformación para desplazar la imagen
                aligned_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=1)

                # Dibujar en el canvas
                scaled_x = int((scaled_x + 32768) / 65535 * _ancho)
                scaled_y = int((scaled_y + 32768) / 65535 * _alto)
                scaled_x = np.clip(scaled_x, 0, _ancho - 1)
                scaled_y = np.clip(scaled_y, 0, _alto - 1)
                cv2.circle(canvas_square, (scaled_x, scaled_y), 5, (0, 0, 255), -1)
                
    cv2.rectangle(canvas_square_2, (scaled_x, scaled_y), (scaled_x + square_size, scaled_y + square_size), (14, 100, 255), -1)                

    
    cv2.imshow("Nose Tracking", canvas_square)
    cv2.imshow("seguimiento", canvas_square_2)
    aligned_frame = cv2.flip(aligned_frame,1)
    aligned_frame =cv2.resize(aligned_frame,(800,600))
    cv2.imshow("Aligned Face", aligned_frame)

cv2.destroyAllWindows()
cap.release()
ui.close()
import numpy as np
import cv2
from capture import HeadTracker
from calibrate import HeadCalibrator
from evdev import UInput, ecodes as e
import evdev
from pynput import keyboard
import threading
capabilities = {
    e.EV_ABS: [
        (e.ABS_RX, evdev.AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)), # Stick derecho X
        (e.ABS_RY, evdev.AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0))  # Stick derecho Y
    ],
    e.EV_KEY: [
        e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y,  # Botones principales
        e.BTN_TL, e.BTN_TR,                   # Gatillos como botones
        e.BTN_SELECT, e.BTN_START, e.BTN_THUMBR,  # Select, Start, Click Stick Derecho
        e.BTN_DPAD_UP, e.BTN_DPAD_DOWN, e.BTN_DPAD_LEFT, e.BTN_DPAD_RIGHT  # D-Pad
    ]
}
ui = UInput(capabilities, name="Control De La Cara")
scaled_x, scaled_y = 0, 0

class HeadMovementDetector:
    def __init__(self, tracker, calibrator):
        self.tracker = tracker
        self.calibrator = calibrator
        self.neutral_position, self.thresholds = calibrator.neutral_position, calibrator.thresholds
        self.directions = {"up": False, "down": False, "left": False, "right": False}
        self.previous_movements = []

    def detect_movement(self, frame):
        """
        Detecta la dirección del movimiento comparando la posición de la nariz con la calibración base.
        """
        nose_pos = self.tracker.get_nose_position(frame)
        if not nose_pos:
            return None, 0, 0  # Devolvemos valores por defecto para evitar errores

        movement = np.array(nose_pos) - self.neutral_position
        movement_x, movement_y = movement

        # Suavizado de movimiento utilizando media móvil
        self.previous_movements.append([movement_x, movement_y])
        if len(self.previous_movements) > 3:  # Usamos los últimos 5 movimientos para suavizar
            self.previous_movements.pop(0)
        
        # Calcular media móvil de los últimos movimientos
        smoothed_movement = np.mean(self.previous_movements, axis=0)
        movement_x, movement_y = smoothed_movement

        # Determinar direcciones en base a umbrales (ajustando para izquierda y derecha)
        self.directions["up"] = movement_y < -self.thresholds[1]
        self.directions["down"] = movement_y > self.thresholds[1]
        self.directions["left"] = movement_x < -self.thresholds[0]  # Izquierda: movimiento negativo
        self.directions["right"] = movement_x > self.thresholds[0]  # Derecha: movimiento positivo

        return self.directions, movement_x, movement_y  # Ahora retorna también los valores suavizados

def map_value(value, in_min, in_max, out_min, out_max):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def send_to_joystick(directions, movement_x, movement_y, neutral_pos=(0, 0), sensitivity=0.45):
    """
    Enviar los movimientos detectados y la velocidad al joystick virtual, con un movimiento
    proporcional a la distancia de la nariz, utilizando los valores máximos y mínimos de cada eje.
    """
    max_value = 32767  # Valor máximo del joystick
    min_value = -32768  # Valor mínimo del joystick
    
    # Configurar el multiplicador para amplificar el movimiento
    multiplier = 50.2 if (movement_x != 0.0 or movement_y != 0.0) else 1.0
    
    

        # Mapear de [-1, 1] a [-32768, 32767]
    # movement_x = map_value(-movement_x, -1, 1, -32768, 32767)
    # movement_y = map_value(movement_y, -1, 1, -32768, 32767)
    # Aplicar el multiplicador y la sensibilidad
    movement_x = int(np.clip(-movement_x * multiplier * sensitivity, min_value, max_value))
    movement_y = int(np.clip(movement_y * multiplier * sensitivity, min_value, max_value))
    
    if movement_x == 0.0 and movement_y == 0.0:
        # Volver a la posición neutral (centro del joystick)
        ui.write(e.EV_ABS, e.ABS_RX, 0)
        ui.write(e.EV_ABS, e.ABS_RY, 0)
    else:
        # Enviar movimientos en la dirección correspondiente
        if directions["up"] or directions["down"]:
            ui.write(e.EV_ABS, e.ABS_RY, movement_y)
        if directions["left"] or directions["right"]:
            ui.write(e.EV_ABS, e.ABS_RX, movement_x)
    
        # Sincronizar los eventos
        ui.syn()

        
        # Sincronizar los eventos
    ui.syn()
    
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


# Función principal
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Asegúrate de usar el ID correcto de la cámara
    tracker = HeadTracker()
    calibrator = HeadCalibrator(tracker)
    calibrator.calibrate(cap)  # Realizar la calibración antes de detectar movimientos
    
    detector = HeadMovementDetector(tracker, calibrator)
    dx, dy = 0.0, 0.0  # Usar flotantes para mayor precisión
    previous_position = [dx, dy]
    movement_threshold = 3  # Umbral mínimo para considerar un movimiento
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar los movimientos de la cabeza
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            scaled_x, scaled_y = 0,0
        directions, movement_x, movement_y = detector.detect_movement(frame)
        
        # if directions == '{\'up\': False, \'down\': False, \'left\': False, \'right\': False}':
        #     scaled_x, scaled_y = 0, 0
        #     print(f"No hay")
        # else:
            # Ignorar movimientos pequeños (menos que el umbral)
        if directions:
            
            if abs(movement_x) > abs(movement_y) and abs(movement_x) > movement_threshold:
                scaled_x, scaled_y = int((movement_x - 0.5)*10.2), 0
            elif abs(movement_y) > abs(movement_x) and abs(movement_y) > movement_threshold:
                scaled_x, scaled_y = 0, int((movement_y - 0.5)*10.2)
            # Enviar la dirección y velocidad al joystick virtual
            send_to_joystick(directions, scaled_x, scaled_y)
        
            

        # Mostrar la imagen con los movimientos detectados
        # cv2.imshow("Head Movement Detection", frame)
        

    cap.release()
    cv2.destroyAllWindows()

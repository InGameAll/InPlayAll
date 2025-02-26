import numpy as np
import pandas as pd
import cv2
from capture import HeadTracker
from calibrate import HeadCalibrator
import time

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
        if len(self.previous_movements) > 5:  # Usamos los últimos 5 movimientos para suavizar
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



# Función para guardar los movimientos detectados en un CSV
def store_movements(movement_data):
    try:
        # Cargar el DataFrame existente si ya existe, o crear uno vacío si no
        try:
            df_existing = pd.read_csv('movement_data.csv')  # Leer los datos existentes
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=['timestamp', 'dx', 'dy', 'speed'])  # Si no existe, crear un DataFrame vacío

        # Usar pd.concat para añadir nuevas filas de movimiento
        df_new = pd.DataFrame(movement_data, columns=['timestamp', 'dx', 'dy', 'speed'])
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)  # Concatenar los DataFrames

        # Guardar los datos actualizados
        df_existing.to_csv('movement_data.csv', index=False)

    except Exception as e:
        print(f"Error al almacenar los movimientos: {e}")


# Función principal
if __name__ == "__main__":
    cap = cv2.VideoCapture(6)  # Asegúrate de usar el ID correcto de la cámara
    tracker = HeadTracker()
    calibrator = HeadCalibrator(tracker)
    calibrator.calibrate(cap)  # Realizar la calibración antes de detectar movimientos

    detector = HeadMovementDetector(tracker, calibrator)

    movement_data = []
    movement_threshold = 2  # Umbral para ignorar pequeños movimientos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # Detectar los movimientos de la cabeza
        directions, movement_x, movement_y = detector.detect_movement(frame)
        
        if directions:
            print(f"Movimientos detectados: {directions}")

            # Ignorar movimientos pequeños (menos que el umbral)
            if abs(movement_x) < movement_threshold and abs(movement_y) < movement_threshold:
                continue

            # Capturar los movimientos de cabeza y calcular la "velocidad"
            speed = np.linalg.norm([movement_x, movement_y])  # Magnitud del movimiento

            # Almacenar los datos de movimiento (timestamp, dx, dy, speed)
            timestamp = time.time()  # Tiempo actual
            movement_data.append([timestamp, movement_x, movement_y, speed])

            # Almacenar los movimientos cada 10 segundos (por ejemplo, para no llenar el archivo rápidamente)
            if len(movement_data) >= 10:
                df = pd.DataFrame(movement_data, columns=['timestamp', 'dx', 'dy', 'speed'])
                store_movements(df)
                movement_data = []  # Limpiar los datos almacenados temporalmente

        # Mostrar la imagen con los movimientos detectados
        cv2.imshow("Head Movement Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

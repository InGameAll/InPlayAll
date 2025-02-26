import numpy as np
import cv2

class HeadCalibrator:
    def __init__(self, tracker):
        self.tracker = tracker
        self.neutral_position = None  # Para almacenar la posición neutral de la nariz
        self.thresholds = None  # Para almacenar los umbrales de movimiento

    def calibrate(self, cap, num_frames=20):
        """
        Calibra la posición neutral de la cabeza y establece los umbrales de movimiento.
        
        cap: objeto VideoCapture de OpenCV que captura los frames de la cámara.
        num_frames: número de frames a capturar para determinar la posición neutral.
        """
        print("Iniciando calibración...")
        
        # Capturamos los primeros frames para definir la posición neutral
        neutral_positions = []
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print("Error al leer el frame.")
                return
            
            nose_pos = self.tracker.get_nose_position(frame)
            if nose_pos:
                neutral_positions.append(nose_pos)
            
            cv2.imshow("Calibración en progreso", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calcular la posición neutral promediando las posiciones detectadas
        self.neutral_position = np.mean(neutral_positions, axis=0).astype(int)
        print(f"Posición neutral de la nariz calibrada: {self.neutral_position}")
        
        # Establecer umbrales de movimiento
        # Los umbrales los definimos como una fracción del rango de movimiento
        self.thresholds = [20, 20]  # Ajusta estos valores según el rango de movimientos esperados
        
        print(f"Umbrales de movimiento establecidos: {self.thresholds}")
        
        # Finalizamos la calibración
        cv2.destroyAllWindows()


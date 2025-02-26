import numpy as np
import cv2
import time
from capture import HeadTracker
from calibrate import HeadCalibrator
from evdev import UInput, AbsInfo, ecodes as e, InputDevice
from pynput import keyboard
import threading
from pynput.mouse import Controller, Button

mouse = Controller()

device_path = "/dev/input/event12"
device = InputDevice(device_path)
sensitivity = 5

capabilities = {
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)),
        (e.ABS_Y, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0))
    ],
    e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_DPAD_UP, e.BTN_X, e.BTN_Y]
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
        self.previous_mouth_state = False  # Estado anterior de la boca

    def detect_movement(self, frame):
        nose_pos = self.tracker.get_nose_position(frame)
        mouth_open = self.tracker.is_mouth_open(frame)  # Nueva detección de boca
        # guinio = self.tracker.is_left_eye_winking(frame)
        
        if mouth_open and not self.previous_mouth_state:
            mouse.click(Button.left, 1)  # Hacer clic izquierdo
        self.previous_mouth_state = mouth_open
        
        # if guinio:
        #     mouse.click(Button.right,1)
        
        if not nose_pos:
            return None, 0, 0

        movement = np.array(nose_pos) - self.neutral_position
        movement_x, movement_y = movement

        self.previous_movements.append([movement_x, movement_y])
        if len(self.previous_movements) > 3:
            self.previous_movements.pop(0)
        
        smoothed_movement = np.mean(self.previous_movements, axis=0)
        movement_x, movement_y = smoothed_movement

        self.directions["up"] = movement_y < -self.thresholds[1]
        self.directions["down"] = movement_y > self.thresholds[1]
        self.directions["left"] = movement_x < -self.thresholds[0]
        self.directions["right"] = movement_x > self.thresholds[0]

        return self.directions, movement_x, movement_y

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

def send_to_joystick(directions, movement_x, movement_y, neutral_pos=(0, 0), sensitivity=0.65):
    max_value, min_value = 32767, -32768
    multiplier = 50.2 if (movement_x != 0.0 or movement_y != 0.0) else 1.0
    movement_x = int(np.clip(movement_x * multiplier * sensitivity, min_value, max_value))
    movement_y = int(np.clip(movement_y * multiplier * sensitivity, min_value, max_value))
    
    if movement_x == 0.0 and movement_y == 0.0:
        ui.write(e.EV_ABS, e.ABS_X, 0)
        ui.write(e.EV_ABS, e.ABS_Y, 0)
        mouse.move(0,0)
    else:
        if directions["up"] or directions["down"]:
            ui.write(e.EV_ABS, e.ABS_Y, movement_y)
        if directions["left"] or directions["right"]:
            ui.write(e.EV_ABS, e.ABS_X, -movement_x)
        
        head_min, head_max = -10000, 10000
        mouse_min, mouse_max = -20, 20
        mouse_dx = int(map_range(-movement_x, head_min, head_max, mouse_min, mouse_max))
        mouse_dy = int(map_range(movement_y, head_min, head_max, mouse_min, mouse_max))
    
        mouse.move((mouse_dx // sensitivity), (mouse_dy // sensitivity))
    ui.syn()

    # Simular el primer toque
    
    

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

def on_press(key):
    try:
        if key.char in key_map:
            ui.write(e.EV_KEY, key_map[key.char], 1)
            ui.syn()
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char in key_map:
            ui.write(e.EV_KEY, key_map[key.char], 0)
            ui.syn()
    except AttributeError:
        pass

key_map = {
    'x': e.BTN_X,
    'y': e.BTN_Y,
    'a': e.BTN_A,
    'b': e.BTN_B
}

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HeadTracker()
    calibrator = HeadCalibrator(tracker)
    calibrator.calibrate(cap)
    
    detector = HeadMovementDetector(tracker, calibrator)
    keyboard_thread = threading.Thread(target=start_keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            scaled_x, scaled_y = 0, 0
        
        directions, movement_x, movement_y = detector.detect_movement(frame)
        if directions:
            send_to_joystick(directions, movement_x, movement_y)
        # frame_resized = cv2.resize(frame, (160, 120))
        # cv2.imshow("Head Movement Detection", frame_resized)
    
    cap.release()
    cv2.destroyAllWindows()
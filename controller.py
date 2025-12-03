import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import threading

# --- Settings ---
CAMERA_ID = 0                         # ID of the camera device
MIN_STABLE_FRAMES = 6                 # Frames required to confirm a gesture
SLIDE_COOLDOWN_SEC = 0.6              # Delay between slide actions
MOUSE_SCALE = 1.5                     # Scale factor for mouse acceleration
CLICK_DIST_THRESHOLD = 0.03           # Distance threshold for click gesture
DRAG_DIST_THRESHOLD = 0.035           # Distance threshold for drag gesture
IS_ACTIVE = True                      # Whether gesture control is active
SMOOTHING_FACTOR = 0.25               # Smoothing factor for cursor movement

# --- MediaPipe & OpenCV Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,          # Continuous tracking mode
    max_num_hands=1,                  # Only track one hand
    min_detection_confidence=0.7,     # Detection confidence
    min_tracking_confidence=0.7       # Tracking confidence
)
cap = cv2.VideoCapture(CAMERA_ID)     # Open camera
if not cap.isOpened():
    print("Cannot open camera.")
    exit(1)

# --- State Variables ---
last_action = 0                        # Timestamp of last slide action
gesture_counter = {"open": 0, "fist": 0, "none": 0}  # Counts stable gestures
is_clicking = False                   # Whether click is happening
is_dragging = False                   # Whether dragging is active
smooth_x = smooth_y = 0               # Smoothed cursor coordinates
initialized = False                   # Whether smoothing init is done
sw, sh = pyautogui.size()             # Screen width & height

# --- Helper Functions ---

# Calculates openness ratio of hand (how open the hand is)
def openness(lm):
    wrist = np.array([lm[0].x, lm[0].y])                # Wrist landmark
    tips_indices = [4, 8, 12, 16, 20]                   # Fingertip indices
    # Distances of fingertips to wrist
    tips_dists = [np.linalg.norm(np.array([lm[i].x, lm[i].y]) - wrist) for i in tips_indices]
    mid_dist = np.linalg.norm(np.array([lm[9].x, lm[9].y]) - wrist)    # Mid-hand point to wrist
    return np.mean(tips_dists) / mid_dist if mid_dist > 1e-6 else 0

# Calculates normalized distance between two landmarks
def distance(lm, i, j):
    p1 = np.array([lm[i].x, lm[i].y])
    p2 = np.array([lm[j].x, lm[j].y])
    return np.linalg.norm(p1 - p2)

# Thread: switches active/inactive mode with 's'
def toggle_active():
    global IS_ACTIVE
    print("Press 's'+Enter to toggle hand control")
    while True:
        try:
            if input().lower() == 's':      # Wait for 's' input
                IS_ACTIVE = not IS_ACTIVE   # Toggle active state
                print("Hand control:", "ACTIVE" if IS_ACTIVE else "INACTIVE")
        except:
            pass

# Start the background thread
threading.Thread(target=toggle_active, daemon=True).start()

# --- Main Loop ---
while True:
    ret, frame = cap.read()                  # Read frame from camera
    if not ret:
        continue

    frame = cv2.flip(frame, 1)               # Mirror image horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = hands.process(rgb_frame)       # Process frame with MediaPipe

    gesture = 'none'                         # Default gesture state

    # Display status (active/inactive)
    status_color = (0, 255, 0) if IS_ACTIVE else (0, 0, 255)
    cv2.putText(frame, f"Active:{'YES' if IS_ACTIVE else 'NO'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark    # Hand landmarks

        # Draw landmarks on frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # --- Gesture Detection ---
        ratio = openness(lm)                             # Hand openness ratio
        gesture = 'open' if ratio > 1.25 else ('fist' if ratio < 0.95 else 'none')

        # --- Cursor Target Calculation ---
        tip = np.array([lm[8].x, lm[8].y])               # Index fingertip position

        # Center proximity: used to adjust movement scaling
        center_prox = 0.5 - abs(tip[0] - 0.5)
        adj_scale = 1 + MOUSE_SCALE * max(0, center_prox - 0.2)

        # Convert normalized coords to screen coords
        tx = np.clip(tip[0] * sw * adj_scale, 0, sw - 1)
        ty = np.clip(tip[1] * sh * adj_scale, 0, sh - 1)

        # --- Smoothing ---
        if not initialized:
            smooth_x, smooth_y = tx, ty
            initialized = True

        # Apply smoothing formula
        smooth_x += (tx - smooth_x) * SMOOTHING_FACTOR
        smooth_y += (ty - smooth_y) * SMOOTHING_FACTOR

        # --- Move Mouse ---
        if IS_ACTIVE:
            try:
                pyautogui.moveTo(round(smooth_x), round(smooth_y))
            except:
                pass

        # --- Click Detection (Thumb → Index distance) ---
        dist_click = distance(lm, 4, 8)

        if IS_ACTIVE and not is_dragging:
            if dist_click < CLICK_DIST_THRESHOLD and not is_clicking:
                pyautogui.click()             # Trigger click
                is_clicking = True
            elif dist_click >= CLICK_DIST_THRESHOLD and is_clicking:
                is_clicking = False

        # --- Drag Detection (Index → Middle finger distance) ---
        dist_drag = distance(lm, 8, 12)

        if IS_ACTIVE:
            if dist_drag < DRAG_DIST_THRESHOLD and not is_dragging:
                is_dragging = True
                pyautogui.mouseDown()         # Start drag
            elif dist_drag >= DRAG_DIST_THRESHOLD and is_dragging:
                is_dragging = False
                pyautogui.mouseUp()           # Stop drag

        # Display drag and click distance on screen
        drag_color = (255, 0, 0) if is_dragging else (0, 255, 255)
        cv2.putText(frame, f"Drag:{'ON' if is_dragging else 'OFF'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, drag_color, 2)
        cv2.putText(frame, f"Click Dist:{dist_click:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        initialized = False               # Reset smoothing when hand disappears

    # --- Stable Gesture Counter ---
    for k in gesture_counter:
        gesture_counter[k] = gesture_counter[k] + 1 if k == gesture else 0

    now = time.time()
    if IS_ACTIVE and now - last_action > SLIDE_COOLDOWN_SEC:
        if gesture_counter['open'] >= MIN_STABLE_FRAMES:
            pyautogui.press('right')     # Slide forward
            last_action = now
            gesture_counter['open'] = 0
        elif gesture_counter['fist'] >= MIN_STABLE_FRAMES:
            pyautogui.press('left')      # Slide backward
            last_action = now
            gesture_counter['fist'] = 0

    # --- Show Window & Exit ---
    cv2.imshow('Hand Gesture Mouse', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Terminated")

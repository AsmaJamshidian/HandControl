# Hand Gesture Mouse Control 🖐️🖥️

Control your computer **entirely with hand gestures** using **OpenCV** and **MediaPipe**. This project allows you to:

* Move the mouse with hand movements
* Click using thumb-to-index gesture
* Drag using index-to-middle finger gesture
* Navigate slides (forward/backward) with open or fist gestures
* Toggle gesture control on/off with `'s'`

---

## 📦 Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

---

## 🎥 Camera Setup

* `CAMERA_ID`: the ID of your webcam (default is `0` for main webcam)

```python
CAMERA_ID = 0
```

---

## ⚙️ Main Settings

| Parameter              | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| `MIN_STABLE_FRAMES`    | Number of consecutive frames required to confirm a gesture (open/fist) |
| `SLIDE_COOLDOWN_SEC`   | Delay between slide actions (seconds)                                  |
| `MOUSE_SCALE`          | Scaling factor for mouse movement                                      |
| `CLICK_DIST_THRESHOLD` | Distance threshold between thumb and index finger to detect click      |
| `DRAG_DIST_THRESHOLD`  | Distance threshold between index and middle finger to detect drag      |
| `IS_ACTIVE`            | Whether gesture control is active                                      |
| `SMOOTHING_FACTOR`     | Mouse movement smoothing factor to avoid jitter                        |

---

## 🖐 How It Works

### 1. Hand Detection

Uses MediaPipe:

```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```

### 2. Hand Openness Calculation

```python
def openness(lm):
    # Calculates how open the hand is based on fingertip distances from wrist
```

### 3. Distance Between Fingers

```python
def distance(lm, i, j):
    # Returns normalized distance between two landmarks
```

### 4. Toggle Gesture Control

* Press `'s'` + Enter in console to toggle between **active/inactive** modes

### 5. Cursor Position

* The **index fingertip** (`lm[8]`) is used as the mouse pointer
* Smoothing is applied using `SMOOTHING_FACTOR`
* Movement scaled with `MOUSE_SCALE` for faster pointer

### 6. Click Detection

* Thumb → Index distance < `CLICK_DIST_THRESHOLD` → click
* Release when distance exceeds threshold

### 7. Drag Detection

* Index → Middle finger distance < `DRAG_DIST_THRESHOLD` → drag
* Release when distance exceeds threshold

### 8. Slide Gestures

* Open hand → Right arrow (forward slide)
* Fist → Left arrow (backward slide)
* Gesture must be stable for at least `MIN_STABLE_FRAMES` frames

---

## 🖥️ On-Screen Display

* Shows whether gesture control is active
* Displays click and drag status
* Shows landmark connections
* Provides distance feedback for debugging

---

## ⌨️ Controls

| Key | Action                             |
| --- | ---------------------------------- |
| `s` | Toggle hand gesture control on/off |
| `q` | Quit the program                   |

---

## ⚡ Notes

* Only **one hand** is tracked (`max_num_hands=1`)
* Low-light conditions may reduce accuracy
* Higher frame rate = smoother cursor movement
* `CLICK_DIST_THRESHOLD` and `DRAG_DIST_THRESHOLD` may need adjustment depending on hand distance from camera

---

## 🚀 Run the Program

```bash
python hand_gesture_mouse.py
```

---

## 📌 Future Improvements

* Support for multiple hands
* Custom gestures for apps or shortcuts
* Record & replay gestures
* Media control (play/pause, volume)

---

## 📸 Preview
<img width="1232" height="715" alt="Screenshot 2025-12-03 095305" src="https://github.com/user-attachments/assets/43e00e45-a95a-4a69-967d-4ab0376f0859" />



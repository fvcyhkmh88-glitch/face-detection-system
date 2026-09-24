# Real-Time Face Detection System

## Introduction
Computer Vision lets computers understand images and video. This mini project
uses a webcam to detect human faces live and mark them with a rectangle. It is
built with Python and OpenCV using the classic **Haar Cascade** method, which
is fast and works on ordinary laptops without a GPU.

## Objective
To build a simple application that:
- starts the webcam,
- detects human faces in real time,
- draws a bounding box around each face,
- shows "Face Detected" and the total number of faces,
- stops the webcam safely.

## Technologies Used
| Technology | Purpose |
|---|---|
| Python 3.8+ | Programming language |
| OpenCV (`opencv-python`) | Webcam access, image processing, drawing |
| Haar Cascade Classifier | Face detection algorithm |
| Webcam | Live video input |

No HTML/CSS is needed; OpenCV's own window is the interface.

## How the System Works
1. The webcam is opened using `cv2.VideoCapture`.
2. A video is just many still images (**frames**). The program reads one frame at a time in a loop.
3. Each frame is converted from colour (BGR) to **grayscale**.
4. The Haar Cascade classifier scans the gray frame with `detectMultiScale()` and returns a list of `(x, y, w, h)` boxes, one per face.
5. `cv2.rectangle()` draws a green box for each face; `cv2.putText()` shows "Face Detected" and the face count.
6. The frame is displayed with `cv2.imshow()`.
7. The loop repeats until the user presses `q` / `ESC` or clicks the STOP button. Then the webcam is released with `cap.release()`.

```
Webcam -> Read Frame -> Grayscale -> Haar Cascade -> Draw Boxes + Text -> Display -> (repeat)
```

## Installation Steps (Windows)
1. Install Python 3.8 or newer from https://www.python.org (tick **"Add Python to PATH"**).
2. Copy this project folder to your computer and open it in Command Prompt:
   ```
   cd path\to\face_detection_project
   ```
3. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## How to Run
```
python main.py
```
- Allow camera access if Windows asks.
- Press **q** or **ESC**, or click the red **STOP** button, to exit.

## Expected Output
- A window showing your live webcam video.
- A **green rectangle** around every face.
- **"Face Detected"** (green) when at least one face is visible, otherwise **"No Face Detected"** (red).
- **"Total Faces: N"** showing the current number of faces.
- A red **STOP** button at the top-left.

## Advantages
- Very simple and easy to understand.
- Runs in real time on a normal laptop CPU.
- Uses only one library (OpenCV); no internet needed after installation.
- Cascade file is bundled with OpenCV, so no extra download.

## Limitations
- Works best on **frontal** faces; side faces may be missed.
- Needs good, even lighting.
- May give occasional false detections or miss small/far faces.
- Cannot recognise *who* the person is, only that a face exists.
- Less accurate than modern deep-learning detectors.

## Future Scope
- Face recognition (identify people) using LBPH or deep learning.
- Attendance system based on detected faces.
- Eye and smile detection with extra cascades.
- Save a photo or log the time when a face appears.
- Upgrade to DNN / MediaPipe detectors for higher accuracy.
- Build a web interface with Flask.

## Conclusion
This project demonstrates the basics of Computer Vision: capturing video,
processing frames, detecting objects with a trained classifier and showing
results live. It is small, works offline and gives a strong foundation for
advanced topics like face recognition and deep learning.

## Troubleshooting
- **"Cannot open the webcam"**: close other apps using the camera, check Windows camera privacy settings, or change `CAMERA_INDEX` to `1` in `main.py`.
- **`pip` not recognised**: reinstall Python with "Add to PATH" ticked, or use `python -m pip install -r requirements.txt`.

## Project Structure
```
face_detection_project/
├── main.py            # main program
├── requirements.txt   # dependencies
└── README.md          # documentation
```
Note: `haarcascade_frontalface_default.xml` is already included inside OpenCV
(`cv2.data.haarcascades`). If you want to keep a copy, place it next to
`main.py` and the program will use it automatically.

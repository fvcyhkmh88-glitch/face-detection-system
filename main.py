"""
Real-Time Face Detection System
--------------------------------
A simple BCA mini project using Python, OpenCV and a Haar Cascade classifier.

What it does:
  1. Opens the webcam.
  2. Detects faces in every video frame.
  3. Draws a green rectangle around each face.
  4. Shows "Face Detected" and the total number of faces.
  5. Stops safely when you press 'q' / ESC or click the STOP button.
"""

import os
import sys
import cv2

# ---------------------------------------------------------------
# Settings (easy to change)
# ---------------------------------------------------------------
CAMERA_INDEX = 0            # 0 = default webcam, 1 = second camera
WINDOW_NAME = "Real-Time Face Detection System"

# STOP button position on screen: (x1, y1, x2, y2)
BUTTON = (10, 10, 110, 50)

# Shared flag that the mouse click handler can change
stop_requested = False


def load_face_detector():
    """Load the Haar Cascade face detector.

    OpenCV ships with this XML file, so we normally do not need to
    download anything. If a copy of the file is placed next to main.py,
    that copy is used instead.
    """
    local_file = "haarcascade_frontalface_default.xml"
    if os.path.exists(local_file):
        path = local_file
    else:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        print("ERROR: Could not load the Haar Cascade file:", path)
        sys.exit(1)
    return detector


def open_webcam():
    """Start the webcam and return the capture object (or exit on failure)."""
    # CAP_DSHOW makes the webcam start faster and more reliably on Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open the webcam.")
        print("Tips: connect a webcam, close other apps using it (Zoom, Teams),")
        print("      and check Windows Settings > Privacy > Camera permissions.")
        sys.exit(1)
    return cap


def mouse_click(event, x, y, flags, param):
    """Called by OpenCV when the mouse is used. Detects a click on STOP."""
    global stop_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            stop_requested = True


def draw_stop_button(frame):
    """Draw a simple red STOP button in the top-left corner."""
    x1, y1, x2, y2 = BUTTON
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), -1)   # filled red box
    cv2.putText(frame, "STOP", (x1 + 18, y2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    face_detector = load_face_detector()
    cap = open_webcam()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_click)

    print("Webcam started. Press 'q' or ESC (or click STOP) to exit.")

    try:
        while not stop_requested:
            # 1. Read one frame from the webcam
            success, frame = cap.read()
            if not success:
                print("ERROR: Could not read a frame from the webcam.")
                break

            # 2. Convert to grayscale (Haar Cascade works on gray images)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 3. Detect faces
            #    scaleFactor  : how much the image is shrunk at each step
            #    minNeighbors : higher = fewer false detections
            #    minSize      : ignore faces smaller than 30x30 pixels
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # 4. Draw a rectangle around every face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 5. Show status text and face count
            face_count = len(faces)
            if face_count > 0:
                cv2.putText(frame, "Face Detected", (130, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face Detected", (130, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(frame, f"Total Faces: {face_count}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            draw_stop_button(frame)

            # 6. Show the frame in a window
            cv2.imshow(WINDOW_NAME, frame)

            # 7. Check keyboard: 'q' or ESC (27) stops the program
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # Also stop if the user closes the window with the X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        # Always release the webcam and close windows, even after an error
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped safely. Goodbye!")


if __name__ == "__main__":
    main()

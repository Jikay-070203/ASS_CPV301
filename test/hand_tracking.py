import cv2
import numpy as np
from skimage.feature import hog
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), 
                              visualize=True)
    return features, hog_image

def detect_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    return mask

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def start_detection(source):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

    elif source == "video":
        video_path = filedialog.askopenfilename(title="Select video file",
                                                filetypes=(("MP4 files", "*.mp4"), 
                                                           ("AVI files", "*.avi"), 
                                                           ("All files", "*.*")))
        if not video_path:
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        hand_mask = detect_hand(frame)
        largest_contour = get_largest_contour(hand_mask)

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            hand_region = preprocessed_frame[y:y+h, x:x+w]

            if hand_region.size > 0:
                hog_features, hog_image = extract_hog_features(hand_region)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Hand Tracking', frame)
                cv2.imshow('HOG Features', hog_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Hand Detection")

    tk.Label(root, text="Choose Source").pack(pady=10)

    tk.Button(root, text="Webcam", command=lambda: start_detection("webcam")).pack(pady=5)
    tk.Button(root, text="Video File", command=lambda: start_detection("video")).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()

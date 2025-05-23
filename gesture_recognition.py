"""
Real-time hand gesture recognition using OpenCV.
"""
import cv2
import numpy as np
import time
from pathlib import Path
import os

class GestureRecognition:
    def __init__(self):
        """Initialize the gesture recognition system."""
        self.cap = cv2.VideoCapture(0)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # Get the camera frame dimensions
        _, frame = self.cap.read()
        if frame is not None:
            frame_h, frame_w = frame.shape[:2]
            # Make ROI larger and more centered
            roi_size = min(500, min(frame_w-100, frame_h-100))
            x_start = (frame_w - roi_size) // 2
            y_start = (frame_h - roi_size) // 2
            self.roi_position = (x_start, y_start, roi_size, roi_size)
        else:
            self.roi_position = (50, 50, 500, 500)  # larger fallback values
        
        # Parameters for hand detection
        self.roi_color = (0, 255, 0)  # Green color for ROI
        
        # Load face cascade classifier
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            print("Warning: Face cascade classifier not loaded!")
        
        # Load reference gestures
        self.reference_gestures = {}
        self.gesture_folders = []  # Store folder names
        self.load_reference_gestures()
        print(f"Loaded {len(self.reference_gestures)} reference gestures")
        
        # Initialize gesture smoothing
        self.last_gestures = []
        self.gesture_smoothing_window = 5
        
        # Create folder list window
        self.create_folder_list_window()

    def create_folder_list_window(self):
        """Create a window showing available gesture folders."""
        # Create a black image for the folder list
        window_height = max(400, len(self.gesture_folders) * 30 + 60)  # Height based on number of folders
        folder_window = np.zeros((window_height, 300, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(folder_window, "Available Gestures", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add folder names
        for i, folder in enumerate(self.gesture_folders):
            y_pos = 60 + i * 30
            cv2.putText(folder_window, f"- {folder}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Create window and show the list
        cv2.namedWindow('Available Gestures', cv2.WINDOW_NORMAL)
        cv2.imshow('Available Gestures', folder_window)

    def load_reference_gestures(self):
        """Load reference gestures from the datasets/hand_gestures directory."""
        base_path = Path("datasets/hand_gestures")
        
        if not base_path.exists():
            print(f"Warning: Reference directory {base_path} not found")
            return

        self.gesture_folders = []  # Clear existing folders
        for gesture_dir in base_path.iterdir():
            if gesture_dir.is_dir():
                gesture_name = gesture_dir.name
                self.gesture_folders.append(gesture_name)  # Add folder name to list
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files.extend(list(gesture_dir.glob(f'*{ext}')))
                
                if image_files:
                    print(f"Found {len(image_files)} images for gesture {gesture_name}")
                    references = []
                    for img_path in image_files:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Convert to grayscale and apply preprocessing
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = cv2.resize(img, (200, 200))
                            img = cv2.equalizeHist(img)  # Improve contrast
                            references.append(img)
                    
                    if references:
                        self.reference_gestures[gesture_name] = references
                        print(f"Loaded {len(references)} references for {gesture_name}")
        
        # Sort folder names alphabetically
        self.gesture_folders.sort()

    def detect_faces(self, frame):
        """Detect faces in the frame and return a mask excluding face regions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        for (x, y, w, h) in faces:
            # Expand face region for better exclusion
            expanded_y1 = max(0, y-30)
            expanded_y2 = min(frame.shape[0], y+h+30)
            expanded_x1 = max(0, x-30)
            expanded_x2 = min(frame.shape[1], x+w+30)
            face_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = 0
            
        return face_mask

    def detect_skin(self, frame, face_mask=None):
        """Enhanced skin detection using multiple color spaces."""
        # Convert to different color spaces
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # More permissive YCrCb range
        min_YCrCb = np.array([0, 130, 75], dtype=np.uint8)
        max_YCrCb = np.array([255, 185, 140], dtype=np.uint8)
        
        # More permissive HSV range
        min_HSV = np.array([0, 15, 60], dtype=np.uint8)
        max_HSV = np.array([25, 255, 255], dtype=np.uint8)
        
        # Create masks
        mask_ycrcb = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
        mask_hsv = cv2.inRange(hsv, min_HSV, max_HSV)
        
        # Combine masks with OR operation for more permissive detection
        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
        
        if face_mask is not None:
            skin_mask = cv2.bitwise_and(skin_mask, face_mask)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=3)  # More dilation
        skin_mask = cv2.GaussianBlur(skin_mask, (7,7), 0)
        
        return skin_mask

    def analyze_hand_shape(self, contour):
        """Analyze hand shape with more permissive criteria."""
        if contour is None or len(contour) < 5:
            return False
            
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # More permissive shape features
        solidity = float(area) / hull_area if hull_area > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        rect_width = min(rect[1])
        rect_height = max(rect[1])
        aspect_ratio = float(rect_width) / rect_height if rect_height > 0 else 0
        
        # More permissive criteria
        if (0.25 < solidity < 0.98 and  # More permissive solidity range
            0.1 < circularity < 0.9 and  # More permissive circularity range
            0.15 < aspect_ratio < 0.9):  # More permissive aspect ratio
            return True
            
        return False

    def find_hand_contour(self, mask):
        """Find the hand contour with improved detection."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:  # Lower minimum area threshold
                continue
                
            if self.analyze_hand_shape(contour):
                return contour
        
        return None

    def compare_gestures(self, current_frame):
        """Compare current frame with reference frames using improved matching."""
        # Preprocess current frame
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.resize(current_gray, (200, 200))
        current_gray = cv2.equalizeHist(current_gray)  # Improve contrast
        
        best_match = None
        best_score = 0
        
        for gesture_name, references in self.reference_gestures.items():
            gesture_score = 0
            for reference in references:
                try:
                    # Try multiple matching methods
                    result1 = cv2.matchTemplate(current_gray, reference, cv2.TM_CCOEFF_NORMED)
                    result2 = cv2.matchTemplate(current_gray, reference, cv2.TM_CCORR_NORMED)
                    
                    # Combine scores
                    score = (np.max(result1) + np.max(result2)) / 2
                    gesture_score = max(gesture_score, score)
                except Exception as e:
                    continue
            
            if gesture_score > best_score:
                best_score = gesture_score
                best_match = gesture_name
        
        return best_match, best_score

    def smooth_gesture_recognition(self, gesture, confidence):
        """Smooth gesture recognition over time."""
        if gesture is None:
            self.last_gestures = []
            return None, 0
            
        self.last_gestures.append((gesture, confidence))
        if len(self.last_gestures) > self.gesture_smoothing_window:
            self.last_gestures.pop(0)
            
        if len(self.last_gestures) < 3:  # Need minimum number of samples
            return None, 0
            
        # Count occurrences of each gesture
        gesture_counts = {}
        for g, c in self.last_gestures:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
        # Find most common gesture
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        if most_common[1] >= len(self.last_gestures) * 0.6:  # 60% threshold
            # Calculate average confidence for this gesture
            avg_confidence = np.mean([c for g, c in self.last_gestures if g == most_common[0]])
            return most_common[0], avg_confidence
            
        return None, 0

    def calculate_fps(self):
        """Calculate and return the FPS."""
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time-self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return int(fps)

    def calculate_optimal_font_scale(self, text, width, height):
        """Calculate the optimal font scale based on text and display area."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        min_scale = 0.1
        max_scale = 4.0
        
        # Binary search for optimal font scale
        while max_scale - min_scale > 0.1:
            current_scale = (min_scale + max_scale) / 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, current_scale, thickness)
            
            # Check if text fits within width and height with some padding
            if text_width < width * 0.9 and text_height < height * 0.8:
                min_scale = current_scale
            else:
                max_scale = current_scale
        
        return min_scale

    def run(self):
        """Run the hand gesture recognition system."""
        print("Hand Gesture Recognition Started")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug view")
        print("Press 'r' to refresh gesture folders")
        
        debug_view = False
        current_gesture = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                
                # Get frame dimensions
                frame_h, frame_w = frame.shape[:2]
                
                # Create gesture display section at the top
                gesture_display = np.zeros((100, frame_w, 3), dtype=np.uint8)
                
                # Detect faces and create face exclusion mask
                face_mask = self.detect_faces(frame)
                
                # Extract ROI
                x, y, w, h = self.roi_position
                roi = frame[y:y+h, x:x+w].copy()
                roi_face_mask = face_mask[y:y+h, x:x+w]
                
                # Detect skin in ROI
                skin_mask = self.detect_skin(roi, roi_face_mask)
                
                # Find hand contour
                hand_contour = self.find_hand_contour(skin_mask)
                
                # Draw ROI rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.roi_color, 2)
                
                # Reset current gesture if no hand detected
                if hand_contour is None:
                    current_gesture = None
                
                if hand_contour is not None:
                    # Create a clean mask for the hand
                    clean_mask = np.zeros_like(skin_mask)
                    cv2.drawContours(clean_mask, [hand_contour], -1, 255, -1)
                    
                    # Extract hand region
                    hand_region = cv2.bitwise_and(roi, roi, mask=clean_mask)
                    
                    # Compare with reference gestures
                    gesture_name, confidence = self.compare_gestures(hand_region)
                    
                    # Apply smoothing
                    smooth_gesture, smooth_confidence = self.smooth_gesture_recognition(gesture_name, confidence)
                    current_gesture = smooth_gesture if smooth_gesture and smooth_confidence > 0.35 else None
                    
                    # Draw the contour
                    cv2.drawContours(roi, [hand_contour], -1, (0, 255, 0), 2)
                    
                    # Draw convex hull
                    hull = cv2.convexHull(hand_contour)
                    cv2.drawContours(roi, [hull], -1, (255, 0, 0), 2)
                    
                    if smooth_gesture and smooth_confidence > 0.35:  # Lower confidence threshold
                        cv2.putText(frame, f'Gesture: {smooth_gesture}', (x, y-30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f'Confidence: {smooth_confidence:.2f}', (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Update gesture display section
                if current_gesture:
                    # Draw background for better visibility
                    cv2.rectangle(gesture_display, (0, 0), (frame_w, 100), (40, 40, 40), -1)
                    # Draw border
                    cv2.rectangle(gesture_display, (0, 0), (frame_w-1, 99), (100, 100, 100), 1)
                    
                    # Calculate optimal font scale for the text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 2
                    font_scale = self.calculate_optimal_font_scale(current_gesture, frame_w, 100)
                    
                    # Calculate text size for centering
                    (text_width, text_height), baseline = cv2.getTextSize(current_gesture, font, font_scale, thickness)
                    text_x = (frame_w - text_width) // 2
                    text_y = (100 + text_height) // 2
                    
                    # Draw text with outline for better visibility
                    cv2.putText(gesture_display, current_gesture, (text_x, text_y), 
                              font, font_scale, (0, 0, 0), thickness + 2)  # outline
                    cv2.putText(gesture_display, current_gesture, (text_x, text_y),
                              font, font_scale, (255, 255, 255), thickness)  # white text
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                cv2.putText(frame, f'FPS: {fps}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

                if debug_view:
                    try:
                        # Convert masks to BGR for visualization
                        skin_mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
                        face_mask_roi_bgr = cv2.cvtColor(roi_face_mask, cv2.COLOR_GRAY2BGR)
                        
                        # Ensure all images have the same dimensions
                        roi_h, roi_w = roi.shape[:2]
                        skin_mask_bgr = cv2.resize(skin_mask_bgr, (roi_w, roi_h))
                        face_mask_roi_bgr = cv2.resize(face_mask_roi_bgr, (roi_w, roi_h))
                        
                        # Create debug display with three panels
                        debug_display = np.zeros((roi_h, roi_w * 3, 3), dtype=np.uint8)
                        debug_display[:, :roi_w] = roi  # Original ROI
                        debug_display[:, roi_w:roi_w*2] = face_mask_roi_bgr  # Face mask
                        debug_display[:, roi_w*2:] = skin_mask_bgr  # Skin mask
                        
                        # Add labels
                        cv2.putText(debug_display, 'Original', (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(debug_display, 'Face Mask', (roi_w + 10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(debug_display, 'Skin Mask', (roi_w*2 + 10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Resize debug display to match frame width
                        debug_display = cv2.resize(debug_display, (frame_w, int(frame_w * debug_display.shape[0] / debug_display.shape[1])))
                        
                        # Create padding if needed to match frame width
                        if debug_display.shape[1] != frame.shape[1]:
                            debug_display = cv2.resize(debug_display, (frame.shape[1], int(frame.shape[1] * debug_display.shape[0] / debug_display.shape[1])))
                        
                        # Combine gesture display, main frame and debug view
                        combined_display = np.vstack([gesture_display, frame, debug_display])
                        cv2.imshow('Hand Gesture Recognition', combined_display)
                    except Exception as e:
                        print(f"Debug view error: {str(e)}")
                        debug_view = False
                else:
                    # Show gesture display and main view
                    combined_display = np.vstack([gesture_display, frame])
                    cv2.imshow('Hand Gesture Recognition', combined_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    debug_view = not debug_view
                elif key == ord('r'):
                    # Refresh gesture folders
                    self.load_reference_gestures()
                    self.create_folder_list_window()
                    
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = GestureRecognition()
        recognizer.run()
    except Exception as e:
        print(f"Error occurred: {str(e)}") 
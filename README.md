# Hand Gesture Recognition System

A real-time hand gesture recognition system using OpenCV that can detect and classify hand gestures based on reference images.

## Requirements

- Python 3.7 - 3.11 (Recommended: Python 3.8 or 3.9)
- Webcam
- The required Python packages are listed in `requirements.txt`

## Python Version Compatibility

This program has been tested with:
- Python 3.7.x
- Python 3.8.x
- Python 3.9.x
- Python 3.10.x
- Python 3.11.x

Note: While newer Python versions might work, we recommend using Python 3.8 or 3.9 for best compatibility.

## Project Structure

```
.
├── gesture_recognition.py    # Main program file
├── requirements.txt         # Python dependencies
├── datasets/               # Directory containing gesture reference images
│   └── hand_gestures/     # Subdirectory for different gesture categories
│       ├── gesture1/      # Each gesture has its own folder
│       ├── gesture2/
│       └── ...
└── README.md              # This file
```

## Setup Instructions

0. Read it:
    look first.md

2. Check your Python version:
   ```bash
   python --version
   ```
   If you don't have Python installed or have a version outside the supported range, download it from python.org

3. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Upgrade pip to the latest version:
   ```bash
   python -m pip install --upgrade pip
   ```

5. Install the required packages:
   ```bash
   # For newer pip versions
   pip install -r requirements.txt

   # If you get errors, try
   pip install --no-cache-dir -r requirements.txt
   ```

6. Prepare your gesture dataset:
   - Create a folder named `datasets/hand_gestures/`
   - Inside it, create a subfolder for each gesture you want to recognize
   - Place reference images (.jpg, .jpeg, or .png) in the corresponding gesture folders

## Running the Program

1. Make sure your webcam is connected and working

2. Run the program:
   ```bash
   python gesture_recognition.py
   ```

## Controls

- Press 'd' to toggle debug view
- Press 'r' to refresh gesture folders
- Press 'q' to quit the program

## Troubleshooting

1. Python Version Issues:
   - If you get syntax errors, make sure you're using a supported Python version
   - Different versions of Python can be installed side by side
   - Use `python --version` to verify which version you're using

2. Package Installation Issues:
   - If pip install fails, try:
     ```bash
     pip install --no-cache-dir -r requirements.txt
     ```
   - For Windows-specific issues:
     ```bash
     pip install --no-cache-dir opencv-python
     pip install --no-cache-dir numpy
     ```

3. Webcam Issues:
   - Make sure your webcam is properly connected
   - Try changing the camera index in the code (default is 0)
   - Check if another application is using the webcam

4. Performance Issues:
   - Ensure proper lighting conditions
   - Keep your hand within the green rectangle
   - Make sure your reference images are clear and well-lit

5. Import Errors:
   - Verify that all requirements are installed: `pip list`
   - Try reinstalling the requirements: `pip install -r requirements.txt --force-reinstall`
   - Make sure you're in the virtual environment (you should see `(venv)` in your terminal)

## Dataset Structure

The program expects reference images to be organized in the following way:
```
datasets/
└── hand_gestures/
    ├── gesture1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── gesture2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

Each gesture folder name will be used as the label for that gesture. 

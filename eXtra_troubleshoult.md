PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML>
 *  History restored 




PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> .\setup.bat
Checking Python version...
Creating virtual environment...
Activating virtual environment...
Upgrading pip...
Collecting pip==25.1.1
  Using cached pip-25.1.1-py3-none-any.whl.metadata (3.6 kB)
Using cached pip-25.1.1-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 25.0.1
    Uninstalling pip-25.0.1:
      Successfully uninstalled pip-25.0.1
Successfully installed pip-25.1.1
Installing dependencies...
Obtaining file:///C:/Users/mypc/Downloads/WORK/PROJECT/SLDUML
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Collecting numpy>=1.26.4 (from gesture_recognition==1.0.0)
  Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting opencv-python>=4.9.0.80 (from gesture_recognition==1.0.0)
  Downloading opencv_python-4.11.0.86-cp37-abi3-win_amd64.whl.metadata (20 kB)
Collecting torch>=2.7.0 (from gesture_recognition==1.0.0)
  Downloading torch-2.7.0-cp313-cp313-win_amd64.whl.metadata (29 kB)
Collecting torchvision>=0.18.0 (from gesture_recognition==1.0.0)
  Downloading torchvision-0.22.0-cp313-cp313-win_amd64.whl.metadata (6.3 kB)
INFO: pip is looking at multiple versions of gesture-recognition to determine which version is compatible with other requirements. This could take a while.
ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11; 1.26.0 Requires-Python <3.13,>=3.9; 1.26.1 Requires-Python <3.13,>=3.9
ERROR: Could not find a version that satisfies the requirement tensorflow>=2.16.1 (from gesture-recognition) (from versions: none)
ERROR: No matching distribution found for tensorflow>=2.16.1
Setup completed successfully
To activate the virtual environment, run: venv\Scripts\activate.bat
To start the application, run: python gesture_recognition.py
Press any key to continue . . .




PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> Set-ExecutionPolicy RemoteSigned -Scope Process
PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> venv\Scripts\activate.bat



PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> Set-ExecutionPolicy RemoteSigned -Scope Process
PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> venv/Scripts/activate 



(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Traceback (most recent call last):
  File "C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML\gesture_recognition.py", line 5, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'






(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> pip install opencv-python mediapipe numpy
Collecting opencv-python
  Using cached opencv_python-4.11.0.86-cp37-abi3-win_amd64.whl.metadata (20 kB)
ERROR: Could not find a version that satisfies the requirement mediapipe (from versions: none)
ERROR: No matching distribution found for mediapipe





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> pip install opencv-python numpy pillow
Collecting opencv-python
  Using cached opencv_python-4.11.0.86-cp37-abi3-win_amd64.whl.metadata (20 kB)
Collecting numpy
  Using cached numpy-2.2.6-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting pillow
  Downloading pillow-11.2.1-cp313-cp313-win_amd64.whl.metadata (9.1 kB)
Downloading opencv_python-4.11.0.86-cp37-abi3-win_amd64.whl (39.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.5/39.5 MB 4.1 MB/s eta 0:00:00
Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl (12.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.6/12.6 MB 6.0 MB/s eta 0:00:00
Downloading pillow-11.2.1-cp313-cp313-win_amd64.whl (2.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 4.6 MB/s eta 0:00:00
Installing collected packages: pillow, numpy, opencv-python
Successfully installed numpy-2.2.6 opencv-python-4.11.0.86 pillow-11.2.1    





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> pip install torch torchvision
Collecting torch
  Using cached torch-2.7.0-cp313-cp313-win_amd64.whl.metadata (29 kB)
Collecting torchvision
  Using cached torchvision-0.22.0-cp313-cp313-win_amd64.whl.metadata (6.3 kB)
Collecting filelock (from torch)
  Downloading filelock-3.18.0-py3-none-any.whl.metadata (2.9 kB)
Collecting typing-extensions>=4.10.0 (from torch)
  Downloading typing_extensions-4.13.2-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy>=1.13.3 (from torch)
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx (from torch)
  Downloading networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch)
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec (from torch)
  Downloading fsspec-2025.5.0-py3-none-any.whl.metadata (11 kB)
Collecting setuptools (from torch)
  Using cached setuptools-80.8.0-py3-none-any.whl.metadata (6.6 kB)
Requirement already satisfied: numpy in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torchvision) (2.2.6)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torchvision) (11.2.1)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch)
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch)
  Downloading MarkupSafe-3.0.2-cp313-cp313-win_amd64.whl.metadata (4.1 kB)
Downloading torch-2.7.0-cp313-cp313-win_amd64.whl (212.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 212.5/212.5 MB 4.0 MB/s eta 0:00:00
Downloading torchvision-0.22.0-cp313-cp313-win_amd64.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 1.8 MB/s eta 0:00:00
Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 1.8 MB/s eta 0:00:00
Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 2.5 MB/s eta 0:00:00
Downloading typing_extensions-4.13.2-py3-none-any.whl (45 kB)
Downloading filelock-3.18.0-py3-none-any.whl (16 kB)
Downloading fsspec-2025.5.0-py3-none-any.whl (196 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading MarkupSafe-3.0.2-cp313-cp313-win_amd64.whl (15 kB)
Downloading networkx-3.4.2-py3-none-any.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 3.1 MB/s eta 0:00:00
Using cached setuptools-80.8.0-py3-none-any.whl (1.2 MB)
Installing collected packages: mpmath, typing-extensions, sympy, setuptools, networkx, MarkupSafe, fsspec, filelock, jinja2, torch, torchvision
Successfully installed MarkupSafe-3.0.2 filelock-3.18.0 fsspec-2025.5.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.4.2 setuptools-80.8.0 sympy-1.14.0 torch-2.7.0 torchvision-0.22.0 typing-extensions-4.13.2






(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> pip install ultralytics
Collecting ultralytics
  Downloading ultralytics-8.3.143-py3-none-any.whl.metadata (37 kB)
Requirement already satisfied: numpy>=1.23.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from ultralytics) (2.2.6)
Collecting matplotlib>=3.3.0 (from ultralytics)
  Downloading matplotlib-3.10.3-cp313-cp313-win_amd64.whl.metadata (11 kB)
Requirement already satisfied: opencv-python>=4.6.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from ultralytics) (4.11.0.86)
Requirement already satisfied: pillow>=7.1.2 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from ultralytics) (11.2.1)
Collecting pyyaml>=5.3.1 (from ultralytics)
  Downloading PyYAML-6.0.2-cp313-cp313-win_amd64.whl.metadata (2.1 kB)
Collecting requests>=2.23.0 (from ultralytics)
  Downloading requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
Collecting scipy>=1.4.1 (from ultralytics)
  Downloading scipy-1.15.3-cp313-cp313-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: torch>=1.8.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from ultralytics) (2.7.0)
Requirement already satisfied: torchvision>=0.9.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from ultralytics) (0.22.0)
Collecting tqdm>=4.64.0 (from ultralytics)
  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting psutil (from ultralytics)
  Downloading psutil-7.0.0-cp37-abi3-win_amd64.whl.metadata (23 kB)
Collecting py-cpuinfo (from ultralytics)
  Downloading py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
Collecting pandas>=1.1.4 (from ultralytics)
  Downloading pandas-2.2.3-cp313-cp313-win_amd64.whl.metadata (19 kB)
Collecting ultralytics-thop>=2.0.0 (from ultralytics)
  Downloading ultralytics_thop-2.0.14-py3-none-any.whl.metadata (9.4 kB)
Collecting contourpy>=1.0.1 (from matplotlib>=3.3.0->ultralytics)
  Downloading contourpy-1.3.2-cp313-cp313-win_amd64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib>=3.3.0->ultralytics)
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib>=3.3.0->ultralytics)
  Downloading fonttools-4.58.0-cp313-cp313-win_amd64.whl.metadata (106 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib>=3.3.0->ultralytics)
  Downloading kiwisolver-1.4.8-cp313-cp313-win_amd64.whl.metadata (6.3 kB)
Collecting packaging>=20.0 (from matplotlib>=3.3.0->ultralytics)
  Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pyparsing>=2.3.1 (from matplotlib>=3.3.0->ultralytics)
  Downloading pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
Collecting python-dateutil>=2.7 (from matplotlib>=3.3.0->ultralytics)
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas>=1.1.4->ultralytics)
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas>=1.1.4->ultralytics)
  Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics)
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting charset-normalizer<4,>=2 (from requests>=2.23.0->ultralytics)
  Downloading charset_normalizer-3.4.2-cp313-cp313-win_amd64.whl.metadata (36 kB)
Collecting idna<4,>=2.5 (from requests>=2.23.0->ultralytics)
  Downloading idna-3.10-py3-none-any.whl.metadata (10 kB)
Collecting urllib3<3,>=1.21.1 (from requests>=2.23.0->ultralytics)
  Downloading urllib3-2.4.0-py3-none-any.whl.metadata (6.5 kB)
Collecting certifi>=2017.4.17 (from requests>=2.23.0->ultralytics)
  Downloading certifi-2025.4.26-py3-none-any.whl.metadata (2.5 kB)
Requirement already satisfied: filelock in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.18.0)
Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (4.13.2)
Requirement already satisfied: sympy>=1.13.3 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (1.14.0)
Requirement already satisfied: networkx in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.4.2)
Requirement already satisfied: jinja2 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.1.6)
Requirement already satisfied: fsspec in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (2025.5.0)
Requirement already satisfied: setuptools in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from torch>=1.8.0->ultralytics) (80.8.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from sympy>=1.13.3->torch>=1.8.0->ultralytics) (1.3.0)
Collecting colorama (from tqdm>=4.64.0->ultralytics)
  Downloading colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\mypc\downloads\work\project\slduml\venv\lib\site-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.2)
Downloading ultralytics-8.3.143-py3-none-any.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 569.7 kB/s eta 0:00:00
Downloading matplotlib-3.10.3-cp313-cp313-win_amd64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 1.4 MB/s eta 0:00:00
Downloading contourpy-1.3.2-cp313-cp313-win_amd64.whl (223 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.58.0-cp313-cp313-win_amd64.whl (2.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 1.5 MB/s eta 0:00:00
Downloading kiwisolver-1.4.8-cp313-cp313-win_amd64.whl (71 kB)
Downloading packaging-25.0-py3-none-any.whl (66 kB)
Downloading pandas-2.2.3-cp313-cp313-win_amd64.whl (11.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.5/11.5 MB 1.7 MB/s eta 0:00:00
Downloading pyparsing-3.2.3-py3-none-any.whl (111 kB)
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading PyYAML-6.0.2-cp313-cp313-win_amd64.whl (156 kB)
Downloading requests-2.32.3-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.2-cp313-cp313-win_amd64.whl (105 kB)
Downloading idna-3.10-py3-none-any.whl (70 kB)
Downloading urllib3-2.4.0-py3-none-any.whl (128 kB)
Downloading certifi-2025.4.26-py3-none-any.whl (159 kB)
Downloading scipy-1.15.3-cp313-cp313-win_amd64.whl (41.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 MB 2.8 MB/s eta 0:00:00
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Downloading ultralytics_thop-2.0.14-py3-none-any.whl (26 kB)
Downloading colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Downloading psutil-7.0.0-cp37-abi3-win_amd64.whl (244 kB)
Downloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Installing collected packages: pytz, py-cpuinfo, urllib3, tzdata, six, scipy, pyyaml, pyparsing, psutil, packaging, kiwisolver, idna, fonttools, cycler, contourpy, colorama, charset-normalizer, certifi, tqdm, requests, python-dateutil, ultralytics-thop, pandas, matplotlib, ultralytics
Successfully installed certifi-2025.4.26 charset-normalizer-3.4.2 colorama-0.4.6 contourpy-1.3.2 cycler-0.12.1 fonttools-4.58.0 idna-3.10 kiwisolver-1.4.8 matplotlib-3.10.3 packaging-25.0 pandas-2.2.3 psutil-7.0.0 py-cpuinfo-9.0.0 pyparsing-3.2.3 python-dateutil-2.9.0.post0 pytz-2025.2 pyyaml-6.0.2 requests-2.32.3 scipy-1.15.3 six-1.17.0 tqdm-4.67.1 tzdata-2025.2 ultralytics-8.3.143 ultralytics-thop-2.0.14 urllib3-2.4.0








(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Creating new Ultralytics Settings v0.0.6 file  
View Ultralytics Settings with 'yolo settings' or at 'C:\Users\mypc\AppData\Roaming\Ultralytics\settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
Starting Hand Gesture Recognition...
Press 'q' to quit
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt'...
100%|███████████████████████████████████████████████████████████████████████████████| 6.25M/6.25M [00:03<00:00, 1.85MB/s]

0: 384x640 (no detections), 197.0ms
Speed: 6.3ms preprocess, 197.0ms inference, 2.7ms postprocess per image at shape (1, 3, 384, 640)
2025-05-23 16:30:14,879 - ERROR - Failed to run gesture recognition: No reference gestures found in dataset
Error occurred: No reference gestures found in dataset





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Hand Gesture Recognition Started
Press 'q' to quit




(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Hand Gesture Recognition Started
Press 'q' to quit





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Loaded reference for Welcome to JKIE
Loaded 1 reference gestures
Hand Gesture Recognition Started
Press 'q' to quit






(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Found 8 images for gesture Welcome to JKIE
Loaded 8 references for Welcome to JKIE
Loaded 1 reference gestures
Hand Gesture Recognition Started
Press 'q' to quit





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Found 8 images for gesture Welcome to JKIE
Loaded 8 references for Welcome to JKIE
Loaded 1 reference gestures
Initializing background subtraction...
Please keep your hand out of view for a few seconds...
Background training complete. You can now show your hand.
Hand Gesture Recognition Started
Press 'q' to quit
Press 'r' to reset background
Resetting background...
Background training complete. You can now show your hand.
Resetting background...
Background training complete. You can now show your hand.
Resetting background...
Background training complete. You can now show your hand.
Resetting background...
Background training complete. You can now show your hand.







(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Found 8 images for gesture Welcome to JKIE
Loaded 8 references for Welcome to JKIE
Loaded 1 reference gestures
Hand Gesture Recognition Started
Press 'q' to quit
Press 'd' to toggle debug view
ERROR:root:Failed to run gesture recognition: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'

Error occurred: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'





(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Found 8 images for gesture Welcome to JKIE
Loaded 8 references for Welcome to JKIE
Loaded 1 reference gestures
Hand Gesture Recognition Started
Press 'q' to quit
Press 'd' to toggle debug view
ERROR:root:Failed to run gesture recognition: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'

Error occurred: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'






(venv) PS C:\Users\mypc\Downloads\WORK\PROJECT\SLDUML> python gesture_recognition.py
Starting Hand Gesture Recognition...
Press 'q' to quit
Found 8 images for gesture Welcome to JKIE
Loaded 8 references for Welcome to JKIE
Loaded 1 reference gestures
Hand Gesture Recognition Started
Press 'q' to quit
Press 'd' to toggle debug view
ERROR:root:Failed to run gesture recognition: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'

Error occurred: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:1261: error: (-27:Null pointer) NULL window: 'Debug View' in function 'cvDestroyWindow'

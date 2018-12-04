# **opencv_python**
js failed me, falling back to ye' olde pyton
 
## Setup:

this link says it all better than i can :P
#### https://solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/

But it is reletively simple setup so below is the main 3 steps:

### 1 - **Download and install python for all users**
#### https://www.python.org/downloads/
Choose a custom install; on the Advanced Options screen make sure to check Install for all users, Add Python to environment variables and Precompile standard library.
<br>
My install uses release 3.7.1 (python-3.7.1-amd64.exe)

### 2 - **Download and install the Numpy version corresponding to your Python installation from:**
#### http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
my install used numpy-1.15.4+mkl-cp37-cp37m-win_amd64.whl

install using: 
```bash
pip install numpy-1.15.4+mkl-cp37-cp37m-win_amd64.whl
```

### 3 - **Download and install the OpenCV version corresponding to your Python installation**
#### http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
my install used opencv_python-3.4.4-cp37-cp37m-win_amd64.whl 

install using: 
```bash
pip install opencv_python-3.4.4-cp37-cp37m-win_amd64.whl
```
<br>

#### To test the install paste the below into a .py file (e.g. `helloworld.py`):
```bash
import cv2 
print(cv2.__version__)
```

then run `helloworld.py` with:
```bash
python helloworld.py
```
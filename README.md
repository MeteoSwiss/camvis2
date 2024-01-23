# CAMVIS2 - Estimation of Visibility on Webcam Images using Multi-Magnification Convolutional Models

## Quickstart
This section provides instruction to build the dataset and replicate the experiments we made. Instructions are provided for UNIX systems only. If you want to use a different OS, you may need to install required libraries manually in order to avoid compatibility issues.
### Overall Setup

#### Enable GPU Acceleration
If you have a NVIDIA GPU with cuda capabilities, it is strongly advised to use it for computations.
You can determine if you already have a GPU driver and its version by executing the following command.
```bash
nvidia-smi
```
If you don't have one yet, you can find an appropriate driver for your graphics card on the [NVIDIA Download Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) webpage so you can speed up your computations.

#### Setup the Environment
In order to install dependencies, it is advised to create a virtual environment. In the instructions, we use [pyenv](https://github.com/pyenv/pyenv). 
You can create an a virtual environment (and enable it) as shown in the command below (replace myenv with the desired name). 
It is advised to use Python 3.10.12 as other versions could lead to compatibility issues. 
```bash
pyenv virtualenv 3.10.12 myenv
pyenv activate myenv
```

#### Install Dependencies
Now that the virtual environment is created, you can install the needed libraries provided in the requirements.txt file as follows : 
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

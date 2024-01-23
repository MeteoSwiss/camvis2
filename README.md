# CAMVIS2 - Estimation of Visibility on Webcam Images using Multi-Magnification Convolutional Networks

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
In order to install dependencies, it is advised to create a virtual environment. We recommand to install and use [pyenv](https://github.com/pyenv/pyenv), although any python environment manager will do.
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

### Dataset Creation
Download the webcam images and the depthmaps from this link *link to come*. 
Uncompress the depth_maps.tar.gz file and put its contents in data/raw/depth_maps. 
Similarly, put the content of the uncompressed images.tar.gz in data/raw/images. 
You can now create the dataset files by executing the command below.
```bash
python data/make_dataset.py
```
In addition to the .gitkeep file, you should now have 5 .pt files and the metadata csv file in the data/processed directory

### Run Experiment
In this experiment, you will train two model architectures using a 9-fold cross validation and compare the results using bootstrapping on the validation performance. One of the model architecture concatenates the features of the patches at different magnification levels, while the other one appends them with scaling so that features are physically aligned from one level to another. Thus, we call this experiment cat_vs_align.

To do train the models and do the inference of the trained models on the valiation sets of each fold, execute the folling script.
```bash
bash train_cat_vs_align.sh
```
Training the 18 models (2 architectures times 9 folds) takes close to 15 hours when using an NVIDIA A100 GPU. 

You can follow the training of the models using tensorboard. To launch a tensorboard session, run the command below in another terminal.
```bash
tensorboard --logdir outputs/training_logs
```

The models checkpoints are saved in the outputs/checkpoints directory.


### Visualize Results
Once model weights are trained, you can infer on the validations sets for each set using each model by executing the command below.
```bash
bash eval_cat_vs_align.sh
```
For each model, you can see the results of the inference on the validation set in subdirectories of outputs/. 
+ Confusion matrices are available in outputs/val_cf_matrices
+ Ground-truth and prediction on webcam images are available in outputs/val_images
+ Histplots of distance vs visibility are available in outputs/val_histplots
+ Bootstrapped performances in terms of loss, accuracy and f1 score are available in outputs/val_scores

Once everything is finished, you can create visualizations of the results using the command below. NB the -v flag will show graphs in a pop-up window. You can remove it if you don't want it.

```bash
python outputs/visualize_scores.py -n cat_vs_align -s -v
```
This will create a subdirectory in outputs/val_scores with the same name as the scores file. Inside the directory, you can find visualization of accuracies, f1 scores and loss values across folds and model architectures.
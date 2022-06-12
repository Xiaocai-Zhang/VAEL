# A Viewpoint Adaptation Ensemble Learning Approach for Vessel Type Recognition with Limited Data
## Setup
Code was developed and tested on Ubuntu 18.04 with Python 3.6 and TensorFlow 2.5.0. You can setup a virtual environment by running the code like this:
```
virtualenv env
source env/bin/activate
cd VAEL
pip3 install -r requirements.txt
```
## Download the DVTR Dataset
Run the following commands to download data sets from Google cloud.
```
gdown https://drive.google.com/uc?id=132b9OeYS_lWbTjYuKXvmqIhPobCAREnq
unzip DVTR
```
## Download the trained generator
Run the following commands to download the trained generator.
```
cd c-wdcgan-gp
gdown https://drive.google.com/uc?id=1u8IDDmBvVMHenREeUkklGdhZ3RskqB7I
unzip save_gen
```
## Download the trained VAEL model
Run the following commands to download the trained VAEL model.
```
cd ..
cd models
gdown https://drive.google.com/uc?id=1lXP8EOSj4HVGa3PAyExQ2h4MQKLZ9H1T
unzip save
```
## Running Models
You can run the following command to replicate the results:
```
python3 vael.py
```

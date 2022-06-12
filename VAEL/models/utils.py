import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from dataset import Dataset
from generate import generate_data
import numpy as np
import cv2
from sklearn.metrics import classification_report,roc_auc_score



def gpu_setting():
    '''
    function to set GPU environment
    :return: None
    '''
    gpu = False
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return None

def set_seed():
    '''
    function to set seed
    :return: None
    '''
    np.random.seed(1329)
    tf.random.set_seed(1329)
    return None

def resize(array, size):
    '''
    function to resize image
    :param array: input img
    :param size: size to be resized, dim x dim
    :return: resized img
    '''
    rsize_li = []
    for img in array:
        rsize_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        rsize_li.append(rsize_img)
    rsize_array = np.array(rsize_li)
    return rsize_array

def get_data(dim,reshape=True):
    '''
    function to get data
    :param dim: dim of img to be ouputted
    :param reshape: reshape img or not
    :return: train, val & test pairs
    '''
    x_train, y_train, x_test, y_test = Dataset().uav_view_dataset()
    x_train_generated, y_train_generated = generate_data().uav_view_sample()

    # scale to [0,1]
    y_train_generated = 0.5 * (y_train_generated + 1)
    x_train = np.concatenate([x_train, x_train_generated], axis=0)
    y_train = np.concatenate([y_train, y_train_generated], axis=0)

    if reshape:
        x_train = resize(x_train, (dim, dim))
        x_test = resize(x_test, (dim, dim))

    # shuffle training data
    np.random.seed(1239)
    x_train = np.random.permutation(x_train)
    np.random.seed(1239)
    y_train = np.random.permutation(y_train)
    # split train and val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    return x_train, x_val, y_train, y_val, x_test, y_test

def train_model(x_train, x_val, y_train, y_val, args, SaveModlFile):
    '''
    function to train the model
    :param x_train: training input
    :param x_val: validation input
    :param y_train: training output
    :param y_val: validation output
    :param args: args
    :param SaveModlFile: path to save model
    :return: None
    '''
    set_seed()
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # pre-training
    model.compile(optimizer=RMSprop(learning_rate=args.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=args.pre_epoch, batch_size=args.batchsize, verbose=1)

    for layer in model.layers[:args.l_f]:
        layer.trainable = False
    for layer in model.layers[args.l_f:]:
        layer.trainable = True

    model.compile(optimizer=RMSprop(learning_rate=args.lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # fine-tuning
    mcp_save = callbacks.ModelCheckpoint(SaveModlFile, save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x=x_train, y=y_train, epochs=args.ft_epoch, batch_size=args.batchsize,
              validation_data=(x_val, y_val), callbacks=[mcp_save], verbose=1)
    return None

def predict(x_test,SaveModlFile):
    '''
    function to predict with a trained model
    :param x_test: test input
    :param SaveModlFile: path of saved model
    :return: predicted output
    '''
    model = load_model(SaveModlFile)
    predictions_test = model.predict(x_test)
    return predictions_test

def ensemble(x_test_1,x_test_2,args):
    '''
    ensemble function
    :param x_test_1: test input for CNN1
    :param x_test_2: test input for CNN2
    :param args: args
    :return: ensemble prediction output
    '''
    predictions_test_1 = predict(x_test_1, args.SaveModlFile_1)
    predictions_test_2 = predict(x_test_2, args.SaveModlFile_2)
    predictions_test = (predictions_test_1 + predictions_test_2) / 2
    return predictions_test

def evaluate(GroundTruth,Prediction):
    '''
    evaluation function
    :param GroundTruth: groundtruth probability distribution
    :param Prediction: prediction probability distribution
    :return: accuray, precision, recall, f1-score & AUC
    '''
    GroundTruth_Idx = np.argmax(GroundTruth, axis=1).tolist()
    Prediction_Idx = np.argmax(Prediction, axis=1).tolist()
    cr = classification_report(GroundTruth_Idx, Prediction_Idx, output_dict=True)
    AUC = roc_auc_score(GroundTruth, Prediction, average='macro')
    return round(cr['accuracy'],4),round(cr['macro avg']['precision'],4),round(cr['macro avg']['recall'],4),round(cr['macro avg']['f1-score'],4),AUC


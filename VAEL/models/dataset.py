import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import random



class config:
    '''
    define parameters & paths
    '''
    train_sample = 50

    save_tanker_source_path = '../DVTR/front-view/TA/'
    save_container_source_path = '../DVTR/front-view/CS/'
    save_bulkcarrier_source_path = '../DVTR/front-view/BC/'
    save_generalcargo_source_path = '../DVTR/front-view/GC/'

    save_tanker_target_path = '../DVTR/UAV-view/TA/'
    save_container_target_path = '../DVTR/UAV-view/CS/'
    save_bulkcarrier_target_path = '../DVTR/UAV-view/BC/'
    save_generalcargo_target_path = '../DVTR/UAV-view/GC/'


# set seed
seed = 409
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)


class Dataset:
    def samplefile(self,file_li):
        '''
        randomly select samples
        :param file_li: original file list
        :return: selected file list & un-selected file list
        '''
        l = list(range(len(file_li)))
        kp_idx = random.sample(l, config.train_sample)
        kp_idx.sort()
        file_li_ = [file_li[idx] for idx in kp_idx]
        file_li_rest = [file for file in file_li if file not in file_li_]
        return file_li_,file_li_rest

    def channel4to3(self,img):
        '''
        reduce 4 channel img to 3 channel
        :param img: 4 channel img
        :return: 3 channel img
        '''
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def rt_data_label(self,dir,label,path,reduce=True):
        '''

        :param dir: data directory
        :param label: data label
        :param path:
        :param reduce:
        :return:
        '''
        all_digits = []
        all_labels = []
        for item in dir:
            img = cv2.imread(path + item, cv2.IMREAD_UNCHANGED)
            if reduce:
                img = self.channel4to3(img)
            all_digits.append(img)
            all_labels.append(label)
        all_digits = np.array(all_digits)
        all_labels = np.array(all_labels)
        return all_digits,all_labels

    def uav_view_dataset(self):
        '''
        get UAV-view image data ready
        :return: data & labels
        '''
        # tanker class
        tanker_dir = os.listdir(config.save_tanker_target_path)
        tanker_dir,tanker_dir_rest = self.samplefile(tanker_dir)
        all_digits_tanker,all_labels_tanker=self.rt_data_label(tanker_dir,0,config.save_tanker_target_path,reduce=False)
        all_digits_tanker_rest, all_labels_tanker_rest = self.rt_data_label(tanker_dir_rest, 0, config.save_tanker_target_path,reduce=False)

        # container class
        container_dir = os.listdir(config.save_container_target_path)
        container_dir,container_dir_rest = self.samplefile(container_dir)
        all_digits_container, all_labels_container = self.rt_data_label(container_dir, 1, config.save_container_target_path,reduce=True)
        all_digits_container_rest, all_labels_container_rest = self.rt_data_label(container_dir_rest, 1,
                                                                            config.save_container_target_path,reduce=True)

        # bulkcarrier class
        bulkcarrier_dir = os.listdir(config.save_bulkcarrier_target_path)
        bulkcarrier_dir,bulkcarrier_dir_rest = self.samplefile(bulkcarrier_dir)
        all_digits_bulkcarrier, all_labels_bulkcarrier = self.rt_data_label(bulkcarrier_dir, 2,
                                                                        config.save_bulkcarrier_target_path,reduce=False)
        all_digits_bulkcarrier_rest, all_labels_bulkcarrier_rest = self.rt_data_label(bulkcarrier_dir_rest, 2,
                                                                                  config.save_bulkcarrier_target_path,reduce=False)

        # general cargo class
        generalcargo_dir = os.listdir(config.save_generalcargo_target_path)
        generalcargo_dir,generalcargo_dir_rest = self.samplefile(generalcargo_dir)
        all_digits_generalcargo, all_labels_generalcargo = self.rt_data_label(generalcargo_dir, 3,
                                                                            config.save_generalcargo_target_path,reduce=True)
        all_digits_generalcargo_rest, all_labels_generalcargo_rest = self.rt_data_label(generalcargo_dir_rest, 3,
                                                                                      config.save_generalcargo_target_path,reduce=True)

        all_digits = np.concatenate(
            [all_digits_tanker, all_digits_container, all_digits_bulkcarrier, all_digits_generalcargo], axis=0)
        all_digits = (all_digits.astype("float32") / 255.0) * 2 - 1

        all_digits_rest = np.concatenate(
            [all_digits_tanker_rest, all_digits_container_rest, all_digits_bulkcarrier_rest, all_digits_generalcargo_rest], axis=0)
        all_digits_rest = (all_digits_rest.astype("float32") / 255.0) * 2 - 1

        all_labels = np.concatenate(
            [all_labels_tanker, all_labels_container, all_labels_bulkcarrier, all_labels_generalcargo], axis=0)
        all_labels = keras.utils.to_categorical(all_labels, 4)

        all_labels_rest = np.concatenate(
            [all_labels_tanker_rest, all_labels_container_rest, all_labels_bulkcarrier_rest, all_labels_generalcargo_rest], axis=0)
        all_labels_rest = keras.utils.to_categorical(all_labels_rest, 4)

        return all_digits,all_labels,all_digits_rest,all_labels_rest

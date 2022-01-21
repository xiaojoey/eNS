import os
import csv
import io
import json
import re
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Flatten, Conv1D, Conv2D, Reshape, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.constraints import unit_norm, max_norm
from matplotlib import pyplot as plt
from data_loader import *
from plot import calculate_pr_recall, plot_pr_recall, save_chart, plot_combined
from buckets import SequenceBuckets
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

def reset_seeds():
    """
        Sets seeds for reproducibility purposes
    """
    seed = 12345
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    print('Setting Seeds')

def test_dx(model_paths, token_path, training_length=1000):
    """
    Loads model for testing and predictions on dx
    Args:
        model_paths - (list) List of model file paths
        token_path - (string) Path to the token file
        training_length- (int) Number of words to be used
    Returns:
        prediction - (np.ndarray) Predictions for each DX
        string_list - (list) Strings used for each DX
        int_list - (list) string_list tokenized
    """
    prediction = None
    attribute_names, string_list = get_dx('../data/noonan_r3.csv')
    string_list = list(string_list)
    int_list, indx_words, word_indx, token_path = string_to_ints(string_list, token_path)
    features = pad_sequences(int_list, maxlen=training_length)
    for model_path in model_paths:
        # summarize model.
        model = load_model(model_path)
        # model.summary()
        # make predictions
        output = model.predict(features, verbose=0)
        if prediction is None:
            prediction = output
        else:
            prediction = np.append(prediction, output, axis=1)
    prediction = np.average(prediction, axis=1)
    return prediction, string_list, int_list

def test_term(model_paths, token_path, training_length=1000):
    """
    Loads model for testing and predictions on terms from token file
    Args:
        model_paths - (list) List of model file paths
        token_path - (string) Path to the token file
        training_length - (int) Number of words to be used
    Returns:
        prediction - (np.ndarray) Predictions for each DX
        string_list - (list) Strings used for each DX
        int_list - (list) string_list tokenized
    """
    tokenizer = None
    prediction = None
    with open(token_path) as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    word_indx = tokenizer.word_index
    string_list = list(word_indx.keys())
    int_list, indx_words, word_indx, token_path = string_to_ints(string_list, token_path)
    features = pad_sequences(int_list, maxlen=training_length)
    for model_path in model_paths:
        # summarize model.
        model = load_model(model_path)
        # model.summary()
        # make predictions
        output = model.predict(features, verbose=0)
        if prediction is None:
            prediction = output
        else:
            prediction = np.append(prediction, output, axis=1)
    prediction = np.average(prediction, axis=1)
    return prediction, string_list, int_list


def make_predict(model_paths, patient_path, token_path, training_length=1000):
    """
    Loads model for and makes prediction on patient cases
    Args:
        model_paths - (list) List of model file paths
        patient_path - (string) Path to the patient file
        token_path - (string) Path to the token file
        training_length - (int) Number of words to be used
    Returns:
        prediction - (np.ndarray) Predictions for each patient
        patient_list - (list) Patient list
        string_list - (list) Strings used for each patient
        int_list - (list) string_list tokenized
    """
    prediction = None
    # load dataset
    patient_array, attribute_names, string_list, patient_list = sort_data(patient_path, 100000)
    int_list, indx_words, word_indx, token_path = string_to_ints(string_list, token_path)
    features = pad_sequences(int_list, maxlen=training_length)
    for model_path in model_paths:
        # summarize model.
        model = load_model(model_path)
        # model.summary()
        # make predictions
        output = model.predict(features, verbose=0)
        if prediction is None:
            prediction = output
        else:
            prediction = np.append(prediction, output, axis=1)
        print("prediction done for %s " %(model_path))
    prediction = np.average(prediction, axis=1)
    return prediction, patient_list, string_list, int_list

def remove_term(model_paths, patient_path, token_path, training_length=1000):
    """
    Loads model for and makes prediction on patient cases
    Args:
        model_paths - (list) List of model file paths
        patient_path - (string) Path to the patient file
        token_path - (string) Path to the token file
        training_length - (int) Number of words to be used
    Returns:
        prediction - (np.ndarray) Predictions for each patient
        patient_list - (list) Patient list
        string_list - (list) Strings used for each patient
    """
    prediction = None
    # load dataset
    patient_array, attribute_names, string_list, patient_list = sort_data(patient_path, 10)
    int_list, indx_words, word_indx, token_path = string_to_ints(string_list, token_path)
    features = pad_sequences(int_list, maxlen=training_length)
    for model_path in model_paths:
        # summarize model.
        model = load_model(model_path)
        # model.summary()
        # make predictions
        output = model.predict(features, verbose=0)
        if prediction is None:
            prediction = output
        else:
            prediction = np.append(prediction, output, axis=1)
    prediction = np.average(prediction, axis=1)
    dict = {}
    of_interest = []
    for i in range(len(patient_list)):
        if prediction[i] > .9:
            of_interest.append(i)
            dict[patient_list[i]] = prediction[i]
    print("looking at %s patients" %(len(of_interest)))
    patients = patient_array[of_interest]
    string_list = []
    removed_list = []
    patient_id_list = []
    for patient in patients:
        patient_features = np.array(patient)
        uniques = set(patient_features[:,3])
        uniques = ["ALL"] + list(uniques)
        patient_id = patient_features[0,0]
        patient_id_list = patient_id_list + [patient_id] * len(uniques)
        removed_list = removed_list + uniques
        for item in uniques:
            patient_dx_string = ""
            for patient_feature in patient_features:
                dx_name = patient_feature[3]
                if dx_name != item:
                    patient_dx_string = patient_dx_string + dx_name + " lineend "
            string_list.append(patient_dx_string)
    int_list, _indx_words, _word_indx, _token_path = string_to_ints(string_list, token_path)
    features = pad_sequences(int_list, maxlen=training_length)

    new_prediction = None
    for model_path in model_paths:
        # summarize model.
        model = load_model(model_path)
        # model.summary()
        # make predictions
        output = model.predict(features, verbose=0)
        if new_prediction is None:
            new_prediction = output
        else:
            new_prediction = np.append(new_prediction, output, axis=1)
    new_prediction = np.average(new_prediction, axis=1)
    return new_prediction, patient_id_list, removed_list

class savePredict(keras.callbacks.Callback):
    """
    Callback to save predictions after each epoch
    """
    def __init__(self, validation_generator, validation_labels, output_path, pic_path, save_plot=False, save_pr_recall=False):
        self.out = []
        self.output_path = output_path
        self.validation_generator = validation_generator
        self.validation_labels = validation_labels
        self.pic_path = pic_path
        self.save_plot = save_plot
        self.save_pr_recall = save_pr_recall

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.validation_generator, verbose=2)
        if self.save_pr_recall:
            analysis = calculate_pr_recall(output, self.validation_labels, threshold=.01)
            analysisdir = "{output_path}_{epochNum}_pr.csv".format(output_path=self.output_path, epochNum=epoch)
            save_chart(analysis, analysisdir)

        valdir = "{output_path}_{epochNum}_val.csv".format(output_path=self.output_path, epochNum=epoch)
        with open(valdir, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
            record_writer = csv.writer(csvfile, delimiter=',')
            attribute_names = ['prediction', 'actual']
            record_writer.writerow(attribute_names)
            for i  in range(len(output)):
                row = [output[i][0], self.validation_labels[i]]
                record_writer.writerow(row)

        if self.save_plot:
            chartdir = "{chartdir}_{epochNum}_pr.pdf".format(chartdir=self.pic_path, epochNum=epoch)
            plot_pr_recall(valdir, chartdir)

def focal_loss_custom(alpha, gamma):
    """
    Custom loss function using focal loss
    Args:
        alpha - (double) sets the alpha value
        gamma - (double) ests the gamma value
    """
    def binary_focal_loss(y_true, y_pred):
        fl = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
        y_true_K = K.ones_like(y_true)
        focal_loss = fl(y_true, y_pred)
        return focal_loss
    return binary_focal_loss


if __name__ == '__main__':
    reset_seeds()

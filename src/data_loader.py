import numpy as np
import os
import csv
import io
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def load_features(noonan_path, non_noonan_path, token_path=None):

    """
    Loads the data
    Args:
        noonan_path - (string) Path to csv file of noonan cases
        noon_noonan_path - (string) Path to csv file of non-noonan causes
        token_path - (string) Path to the tokenizer file
    Returns:
        n_features - (np.ndarray) Numpy array of Noonan features
        nn_features - (np.ndarray) Numpy array of non-Noonan features
        indx_words - (dict) Dict for converting ints to words
        word_indx - (dict) Dict for converting words to ints
    """

    """
    first: use sort_data to load the noonan and noonan cases
    """
    n_patient_array, n_attribute_names, n_string_list, n_list = sort_data(noonan_path, 200)
    nn_patient_array, nn_attribute_names, nn_string_list, nn_list = sort_data(non_noonan_path, None)


    """
    second: use string_to_ints data to convert cases to ints
    """
    if token_path is None:
        all_int_list, indx_words, word_indx, token_path = string_to_ints(n_string_list + nn_string_list)
    n_int_list, indx_words, word_indx, token_path = string_to_ints(n_string_list, token_path)
    nn_int_list, indx_words, word_indx, token_path = string_to_ints(nn_string_list, token_path)

    """
    third: create feature and label arrays
    """
    n_features = []
    nn_features = []

    n_features = np.array(n_int_list, dtype=object)
    nn_features = np.array(nn_int_list, dtype=object)

    #stopping shuffling
    # np.random.seed(222)
    np.random.shuffle(nn_features)
    np.random.shuffle(n_features)


    return n_features, nn_features, indx_words, word_indx

def load_from_ids(noonan_path, non_noonan_path, token_path, noonan_id_path, control_id_path, test_path):
    """
    Loads the data from presorted patient ids
    """

    """
    first: use sort_data to load the noonan and noonan cases
    """
    n_patient_array, n_attribute_names, n_string_list, n_list = sort_data(noonan_path, 200)
    nn_patient_array, nn_attribute_names, nn_string_list, nn_list = sort_data(non_noonan_path, None)


    """
    second: use string_to_ints data to convert cases to ints
    """
    if token_path is None:
        all_int_list, indx_words, word_indx, token_path = string_to_ints(n_string_list + nn_string_list)
    n_int_list, indx_words, word_indx, token_path = string_to_ints(n_string_list, token_path)
    nn_int_list, indx_words, word_indx, token_path = string_to_ints(nn_string_list, token_path)

    noonan_zipped = list(zip(n_int_list, n_list))
    non_noonan_zipped = list(zip(nn_int_list, nn_list))

    df_noonan = pd.DataFrame(noonan_zipped, columns=["sequence", "id"])
    df_non_noonan = pd.DataFrame(non_noonan_zipped, columns=["sequence", "id"])
    df_noonan['id'] = pd.to_numeric(df_noonan["id"])
    df_non_noonan['id'] = pd.to_numeric(df_non_noonan["id"])
    print('converted to ints')

    """
    third: load the labeled ids
    """
    noonan_id_df = pd.read_csv(noonan_id_path)
    non_noonan_id_df = pd.read_csv(control_id_path)
    test_id_df = pd.read_csv(test_path)

    """
    fourth: merge the ids with the ints
    """
    noonan_train_df = df_noonan.merge(noonan_id_df, on='id')
    non_noonan_train_df = df_non_noonan.merge(non_noonan_id_df, on='id')
    test_noonan_df = df_noonan.merge(test_id_df, on='id')
    test_non_noonan_df = df_non_noonan.merge(test_id_df, on='id')
    print('merged tables')

    """
    fifth: convert to features and labels and k_fold
    """
    noonan_folds = new_kfold(7, noonan_train_df, test_noonan_df)
    non_noonan_folds = new_kfold(7, non_noonan_train_df, test_non_noonan_df)

    noonan_features = np.append(np.array(noonan_train_df['sequence']), np.array(test_noonan_df['sequence']))
    non_noonan_features = np.append(np.array(non_noonan_train_df['sequence']), np.array(test_non_noonan_df['sequence']))

    return noonan_features, non_noonan_features, noonan_folds, non_noonan_folds, indx_words

def k_fold(k, data_length):
    """
    Returns indices for training, validation, and testing data
    Args:
        k - (int) Number of folds, minimum 3
        data_length - (int) Total number of items in data
    Returns:
        train - (list)
        validation - (list)
        test - (list)
    """
    assert k >= 3
    train = []
    validation = []
    test = []
    fold_length = int(data_length / k)
    folds = []
    for i in range(k):
        start = i * fold_length
        end = (i + 1) * fold_length
        if (i + 1 >= k):
            end = data_length
        folds.append(list(range(start, end)))
    for i in range(k - 1):
        validation.append(folds[i])
        # if (i + 1) > (k - 1):
        #     testfold = (i + 1) - k
        # else:
        #     testfold = i + 1
        testfold = k - 1
        test.append(folds[testfold])
        train_fold = []
        for x in range(k - 1):
            if x != i and x != testfold:
                train_fold += folds[x]
        train.append(train_fold)

    train.append(train[-1] + validation[-1])
    validation.append(test[-1])
    test.append(test[-1])

    return train, validation, test

def new_kfold(k, train_df, test_df):
    """
    Returns indices for training, validation, and testing data for the load_from_ids function
    Args:
        k - (int) Number of folds
        train_df - (df) dataframe with the training data and the fold number of each sample
        test_df - (df) dataframe with the test data and the fold number of each sample
    Returns:
        train - (list)
        validation - (list)
        test - (list)
    """

    folds = []
    for i in range(k):
        folds.append([])
    for index, row in train_df.iterrows():
        folds[row['fold'] - 1].append(index)
    for index, row in test_df.iterrows():
        folds[k - 1].append(index + len(train_df['id']))

    train = []
    validation = []
    test = []
    testfold = k - 1
    for i in range(k - 1):
        validation.append(folds[i])
        test.append(folds[testfold])
        train_fold = []
        for x in range(k - 1):
            if x != i and x != testfold:
                train_fold += folds[x]
        train.append(train_fold)

    train.append(train[-1] + validation[-1])
    validation.append(test[-1])
    test.append(test[-1])
    return train, validation, test

def under_sample(train, validation, test):
    """
    Random under sampler for training, validation, and testing sets
    Args:
        train - (np.ndarray) training dataset
        validation - (np.ndarray) validation dataset
        test - (np.ndarray) testing dataset
    """
    ros = RandomUnderSampler(random_state=0, sampling_strategy=0.001)
    x_validation, y_validation = ros.fit_resample(validation[0], validation[1])
    x_test, y_test = ros.fit_resample(test[0], test[1])

    ros = RandomUnderSampler(random_state=0, sampling_strategy=0.01)
    x_train, y_train = ros.fit_resample(train[0], train[1])

    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    y_test = np.array(y_test)

    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(x_validation.shape[0])
    np.random.shuffle(indices)
    x_validation = x_validation[indices]
    y_validation = y_validation[indices]

    indices = np.arange(x_test.shape[0])
    np.random.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    return((x_train, y_train), (x_validation, y_validation), (x_test, y_test))

def sort_data(path, num_examples=None):
    """
    Sorts the data based on patient number
    Args:
        path - (string) Path to csv file
        num_examples - (int) Number of samples to load
    Returns:
        features - (np.ndarray) numpy array containing all the information of each patient
        attribute_names - (np.ndarray) numpy array containing the headers
        string_list - (list) list of strings descriptions
        patient_list - (list) list of patient ids
    """
    attribute_names = []
    data = []
    string_list = []
    patient_list = []
    prev_id = None
    patients_loaded = 0
    with open(path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            row_id = row[0]
            dx_name = row[3]
            gender = row[4]
            if prev_id == None:
                new_patient = []
                patient_dx_string = gender + " lineend "
                patients_loaded += 1
            elif prev_id != row_id:
                if num_examples != None and patients_loaded >= num_examples:
                    break
                data.append(new_patient)
                string_list.append(patient_dx_string)
                patient_list.append(prev_id)
                new_patient = []
                patient_dx_string = gender + " lineend "
                patients_loaded += 1
            new_patient.append(row)
            patient_dx_string = patient_dx_string + dx_name + " lineend "
            prev_id = row_id
        data.append(new_patient)
        string_list.append(patient_dx_string)
        patient_list.append(prev_id)
    print('{file} read'.format(file=path))
    return np.array(data, dtype=object), np.array(attribute_names), string_list, patient_list

def get_dx(path):
    """
    Retrieve all the unique dx in a file
    Args:
        path - (string) Path to csv file
    Returns:
        attribute_names - (np.ndarray) numpy array containing the headers
        string_set - (set) set of unique strings descriptions
    """
    attribute_names = []
    string_set = set()
    with open(path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            dx_name = row[3]
            patient_dx_string = dx_name + " lineend "
            string_set.add(patient_dx_string)
    return np.array(attribute_names), string_set


def string_to_ints(string_list, token=None, shuffle=False):
    """
    Takes a list of strings and converts them to ints
    Args:
        string_list - (list) List of strings
        token - (string) Path to prefitted token
        shuffle - (bool) Whether to shuffle the text in each sequence
    Returns:
        sequence - (list) List of list of ints
        indx_word - (dict) Dict for converting ints to words
        word_index - (dict) Dict for converting words to ints
        file_path - (string) Path to token
    """
    if token == None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='.,#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower = True, split = ' ')
        tokenizer.fit_on_texts(string_list)
        tokenizer_json = tokenizer.to_json()
        file_path = '../tokenizer/token' + str(len(string_list)) + '.json'
        with io.open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    else:
        file_path = token
        with open(file_path) as f:
            data = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
    sequence = tokenizer.texts_to_sequences(string_list)
    # optional shuffleing of sequences
    if shuffle:
        for seq in sequence:
            np.random.shuffle(seq)
    idx_word = tokenizer.index_word
    word_index = tokenizer.word_index
    return sequence, idx_word, word_index, file_path


def get_noonan_patients(path, noonan_path, normal_path):
    """
    Extracts all the noonan syndrom patient causes
    Args:
        path - (string) Path to csv file
        noonan_path - (string) Where to save the noonan patients
        normal_path - (string) Where to save the non_noonan patients
    Returns:
    """
    attribute_names = []
    diagnosed_patients = []
    with open(path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            row_id = row[0]
            diagnose_text = row[3]
            if "noonan" in diagnose_text.lower():
                diagnosed_patients.append(row_id)
    diagnosed_patients_set = set(diagnosed_patients)
    data_write = []
    non_noonan_write = []
    with open(path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            row_id = row[0]
            write_row = row
            if row_id in diagnosed_patients_set:
                data_write.append(write_row)
            else:
                non_noonan_write.append(write_row)
    with open(noonan_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row  in data_write:
            record_writer.writerow(row)
    with open(normal_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row  in non_noonan_write:
            record_writer.writerow(row)

def remove_noonan(in_path, out_path):
    """
    Removes the descriptions containing the word noonan
    Args:
        in_path - (string) Path to the input file
        out_path - (string) Path to the output file
    Returns:
    """
    data_write = []
    attribute_names = None
    with open(in_path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            row_dx = row[3].lower()
            if 'noonan' not in row_dx:
                data_write.append(row)

    with open(out_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row  in data_write:
            record_writer.writerow(row)

def remove_dx(in_path, out_path, term):
    """
    Removes the icd10 containing the target term
    Args:
        in_path - (string) Path to the input file
        out_path - (string) Path to the output file
        term - (string) Term to be removed
    Returns:
    """
    data_write = []
    attribute_names = None
    with open(in_path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            row_id = row[2].lower()
            if term.lower() not in row_id:
                data_write.append(row)

    with open(out_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row  in data_write:
            record_writer.writerow(row)


def convert_file(in_path, out_path):
    """
    Converts text file to csv file
    Args:
        in_path - (string) Path to the input file
        out_path - (string) Path to the output file
    Returns:
    """
    data_write = []
    attribute_names = None
    with open(in_path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter='|')
        attribute_names = next(record_reader)
        attribute_names = attribute_names[:4] + [attribute_names[-1]]
        for row in record_reader:
            write_row = row[:4] + [row[-1]]
            data_write.append(write_row)

    with open(out_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row in data_write:
            record_writer.writerow(row)

def separate_file(ref_path, targ_path, out_path):
    """
    Finds all patients from the target file that are not in the reference file
    Args:
        ref_path - (string) Path to the reference file
        targ_path - (string) Path to the target file
        out_path - (string) Path to the output file
    Returns:
    """
    data_write = []
    id_list = set()
    attribute_names = None
    with open(ref_path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            id = row[0]
            id_list.add(id)

    with open(targ_path, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            id = row[0]
            if id not in id_list:
                data_write.append(row)

    with open(out_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row in data_write:
            record_writer.writerow(row)

def join_files(path_one, path_two, out_path):
    """
    Joins all the patients in two files without duplicate
    Args:
        path_one - (string) Path to the first file
        path_two - (string) Path to the second file
        out_path - (string) Path to the output file
    Returns:
    """
    data_write = []
    id_list = set()
    id_list2 = set()
    attribute_names = None
    with open(path_one, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            id = row[0]
            id_list.add(id)
            write_row = row[:4] + [row[-1]]
            data_write.append(write_row)

    with open(path_two, newline='\n', encoding="ISO-8859-1") as csvfile:
        record_reader = csv.reader(csvfile, delimiter=',')
        attribute_names = next(record_reader)
        for row in record_reader:
            id = row[0]
            if id not in id_list:
                write_row = row[:4] + [row[-1]]
                data_write.append(write_row)
                id_list2.add(id)

    with open(out_path, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        record_writer.writerow(attribute_names)
        for row in data_write:
            record_writer.writerow(row)

    print(len(id_list) + len(id_list2))

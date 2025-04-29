import csv
import os

import numpy as np
import scipy.io as sio


def add_entry(current_info, save_mode=[], **kwargs):
    for key, value in kwargs.items():
        current_info[key] = {'value': value, 'save_mode': save_mode}


def append_entry(info, save_mode=[], **kwargs):
    for key, value in kwargs.items():
        if key in info:
            info[key] = {'value': np.vstack([info[key]['value'], value]), 'save_mode': save_mode}
        else:
            info[key] = {'value': np.asarray([value]), 'save_mode': save_mode}  # Create first row


def get_positions(info, populations, save_mode):
    for pop in populations:
        for d in range(pop.state_dim):
            col_name = str(pop.id + '_state' + str(d))
            value = pop.x[:, d]
            append_entry(info, save_mode, **{col_name: value})


def print_log(current_info):
    """
    Function to print information
    """
    for key, value in current_info.items():
        if 'print' in value['save_mode']:
            print(f"{key}: {value['value']}; ", end=" ")
    print('\n')


def append_txt(log_name, current_info):
    with open(log_name, mode="a") as txtfile:
        txtfile.write("\n")
        for key, value in current_info.items():
            if 'txt' in value['save_mode']:
                txtfile.write(f"{key}: {value['value']}\n")


def append_csv(log_name, current_info):
    current_info_csv = {}
    for key, value in current_info.items():
        if 'csv' in value['save_mode']:
            if isinstance(value['value'], np.ndarray):
                current_info_csv.update({key: value['value'].tolist()})
            else:
                current_info_csv.update({key: value['value']})
    with open(log_name, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=current_info_csv.keys())
        if csvfile.tell() == 0:  # Write header if the file is empty
            writer.writeheader()
        writer.writerow(current_info_csv)


def save_npz(log_name, data):
    """
    A function to save multiple tensors in a single .npz file to store relevant data.

    Parameters
    ----------
        log_name: string of the .npz output name
        data: dict of {name_tensor: value} where name_tensor is a string and value is np.array

    Returns
    -------
        None

    Notes
    -------
    Create numpy arrays of the data to save and add them to a dictionary. Once saved, access the .npz with the field corresponding to the name used.

    Examples
    -------
    settling_times = np.array([1,2,3,4,5])
    control_efforts = np.array([6,7,8,9,10])
    data = {'settling_time':settling_times, 'control_efforts':control_efforts}
    logger.save_data('my_data.npz', data)

    data_loaded = np.load('path\\my_data.npz')
    settling_times_ = data_loaded['settling_times']
    control_efforts_  = data_loaded['control_efforts']

    """
    npz_data = {}
    for key, value in data.items():
        if 'npz' in value['save_mode']:
            npz_data.update({key: value['value']})
    np.savez(log_name, **npz_data)


def save_mat(log_name, data):
    mat_data = {}
    for key, value in data.items():
        if 'mat' in value['save_mode']:
            mat_data.update({key: value['value']})
    sio.savemat(log_name, mat_data)

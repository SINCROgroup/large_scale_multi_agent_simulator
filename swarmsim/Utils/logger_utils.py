import csv
import numpy as np


def add_entry(current_info, **kwargs):
    current_info.update(kwargs)


def append_txt(log_name, current_info):
    with open(log_name, mode="a") as txtfile:
        txtfile.write("\n")
        for key, value in current_info.items():
            txtfile.write(f"{key}: {value}\n")


def append_csv(log_name, current_info):
    with open(log_name, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=current_info.keys())
        if csvfile.tell() == 0:  # Write header if the file is empty
            writer.writeheader()
        writer.writerow(current_info)

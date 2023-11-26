import os

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def min_max_scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.5  # Handle division by zero (if min_val == max_val)
    return (value - min_val) / (max_val - min_val)
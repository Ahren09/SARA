import json
import os
import random
from datetime import datetime

import numpy as np


def check_cwd():
    """
    Checks if the current working directory is the repository root directory (SARA/)
    """
    basename = os.path.basename(os.path.normpath(os.getcwd()))
    assert basename.lower() in [
        "sara"], "Please run this file from the repository root directory (SARA/)"


def set_seed(seed: int = 42, use_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    
    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
        except ImportError:
            print("Fail to import torch. Skipping torch seed setting")


def project_setup():
    check_cwd()
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)


def print_file_timestamps(file_path):
    """
    Prints the created and last modified timestamps of a file in the OS time zone.

    :param file_path: Path to the file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Get the created and last modified timestamps
    created_timestamp = os.path.getctime(file_path)
    modified_timestamp = os.path.getmtime(file_path)

    # Convert timestamps to local time
    created_time = datetime.fromtimestamp(created_timestamp)
    modified_time = datetime.fromtimestamp(modified_timestamp)

    print(f"\tFile: {file_path}")
    print(f"\t* Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\t* Last Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return created_time, modified_time


def make_json_serializable(record):
    """Convert numpy arrays in record to lists for JSON serialization."""
    return json.loads(json.dumps(record, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))

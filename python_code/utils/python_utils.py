import os
import pickle as pkl
from typing import Tuple

import numpy as np

from dir_definitions import PLOTS_DIR


def save_pkl(pkls_path: str, array: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)


if __name__ == "__main__":
    method_name = 'LCVA'
    info_lengths = [13, 15, 30, 50]
    code_lengths = [87, 93, 138, 198]
    BER_results = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
    FER_results = [[2.5e-01, 8e-02, 2e-02, 1.3e-03, 5e-05],
                   [2.5e-01, 8e-02, 1.8e-02, 1.3e-03, 4e-05],
                   [0.4, 0.07, 1.3e-02, 9e-04, 2.8e-05],
                   [0.3, 0.1, 1.2e-02, 8e-04, 2.8e-05]]
    iterations = 24
    for info_length, code_length, ber_result, fer_result in zip(info_lengths, code_lengths, BER_results, FER_results):
        name = '_'.join([method_name, str(code_length), str(info_length)])
        plots_path = os.path.join(PLOTS_DIR, name + '.pkl')
        tupled_values = (np.array(ber_result), np.array(fer_result), iterations * np.ones(len(fer_result)))
        save_pkl(plots_path, tupled_values)

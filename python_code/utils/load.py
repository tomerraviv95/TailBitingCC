import torch
from scipy import io
from dir_definitions import LTE_TBCC_MAT_PATH
from typing import Tuple
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_code_parameters(code_length: int, crc_length: int, info_length: int) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the generator and parity check matrices of the code
    """
    matrices_path = os.path.join(LTE_TBCC_MAT_PATH,
                                 '_'.join(['LTE_TBCC', str(code_length), str(info_length), str(crc_length)]))
    mat = io.loadmat(matrices_path + '.mat')
    code_gm_inner = mat['Gcode']
    code_gm_outer = mat['Gcrcsys']
    code_pcm = mat['H']
    code_h_outer = np.concatenate([code_gm_outer[:, info_length:].T, np.eye(crc_length)], axis=1)
    code_pcm = torch.Tensor(code_pcm).to(device=device)
    code_gm_inner = torch.Tensor(code_gm_inner).to(device=device)
    code_gm_outer = torch.Tensor(code_gm_outer).to(device=device)
    code_h_outer = torch.Tensor(code_h_outer).to(device=device)

    return code_pcm, code_gm_inner, code_gm_outer, code_h_outer

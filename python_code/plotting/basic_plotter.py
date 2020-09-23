import re
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import os
import numpy as np
import yaml
from python_code.trainers.CVA.cva_trainer import CVATrainer
from python_code.trainers.GatedWCVAE.gated_wcvae_trainer import GatedWCVAETrainer
from python_code.trainers.WCVAE.wcvae_trainer import WCVAETrainer
from python_code.trainers.trainer import Trainer
from python_code.utils.python_utils import load_pkl, save_pkl
from dir_definitions import FIGURES_DIR, PLOTS_DIR, CONFIG_PATH

POLY_NUM = 3
BOTTOM_LIM = 1e-5

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [8.2, 6.45]
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'x-rep CVA': '#0F9D58',  # green
               'WCVAE': 'cyan',
               'Gated WCVAE': '#4285F4',  # blue
               'LCVA': 'orange',
               'LGVA': '#DB4437',  # red
               }
MARKERS_DICT = {'x-rep CVA': 'o',  # green
                'Gated WCVAE': 'D',  # blue
                'WCVAE': '*',
                'LCVA': 's',  # orange
                'LGVA': 'x',  # red
                }
METRICS = ['BER', 'FER', 'VA  runs']


class BasicPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot(method_name: str, snr_range: np.ndarray, metric: str, current_plot: np.ndarray):
        plt.plot(snr_range[:len(current_plot)], current_plot, label=method_name,
                 color=COLORS_DICT[re.sub("\d", "x", method_name)], linewidth=2.2,
                 marker=MARKERS_DICT[re.sub("\d", "x", method_name)], linestyle='-.', markersize=12)
        if metric in ['BER', 'FER']:
            plt.yscale('log')
        plt.ylabel(metric)
        plt.xlabel('$E_b/N_0$ [dB]')
        plt.grid(which='both', ls='--')
        plt.xlim([snr_range[0] - 0.1, snr_range[-1] + 0.1])
        if metric in ['BER', 'FER']:
            plt.legend(loc='lower left', prop={'size': 15})
            plt.ylim(bottom=BOTTOM_LIM, top=1)
        else:
            pass

    @staticmethod
    def get_plots(method_name: str, trainer: Trainer, run_over: bool, info_length: int, code_length: int) -> Dict[
        str, np.ndarray]:

        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_name = '_'.join([method_name, str(code_length), str(info_length)])
        plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')

        if os.path.isfile(plots_path) and not run_over:
            print("Loading plots")
            to_plot_curves = load_pkl(plots_path)
        else:
            print("calculating fresh")
            to_plot_curves = trainer.evaluate()
            save_pkl(plots_path, to_plot_curves)

        current_plots_dict = {}
        for i, metric in enumerate(METRICS):
            current_plots_dict[metric] = to_plot_curves[i]
        return current_plots_dict


if __name__ == '__main__':
    basic_plotter = BasicPlotter()
    run_over = False
    info_lengths = [30]
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    methods_names = ['3-rep CVA',
                     'LCVA',
                     'Gated WCVAE',
                     'WCVAE',
                     'LGVA']
    snr_range = np.linspace(config['val_SNR_start'], config['val_SNR_end'], num=config['val_num_SNR'])

    for info_length in info_lengths:
        print(f'info num - {info_length}')
        code_length = POLY_NUM * (info_length + config['crc_length'])
        method_name_to_trainer_dict = {'3-rep CVA': CVATrainer(run_name=None,
                                                               load_from_checkpoint=False,
                                                               info_length=info_length,
                                                               code_length=code_length),
                                       'Gated WCVAE': GatedWCVAETrainer(
                                           run_name=f'gated_wcvae_{info_length}',
                                           load_from_checkpoint=True,
                                           info_length=info_length,
                                           code_length=code_length),
                                       'WCVAE': WCVAETrainer(run_name=f'gated_wcvae_{info_length}',
                                                             load_from_checkpoint=True,
                                                             info_length=info_length,
                                                             code_length=code_length),
                                       'LCVA': None,
                                       'LGVA': None}
        plots_dict = {}
        for method_name in methods_names:
            print(method_name)
            trainer = method_name_to_trainer_dict[method_name]
            plots_dict[method_name] = basic_plotter.get_plots(method_name, trainer, run_over, info_length, code_length)

        current_day_time = datetime.datetime.now()
        folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}_{info_length}'
        if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
            os.makedirs(os.path.join(FIGURES_DIR, folder_name))

        for metric in METRICS:
            plt.figure()
            for method_name in methods_names:
                current_plot = plots_dict[method_name][metric]
                basic_plotter.plot(method_name, snr_range, metric, current_plot)
            plt.savefig(os.path.join(FIGURES_DIR, folder_name, metric + '.png'), bbox_inches='tight')

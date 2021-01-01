import re

from dir_definitions import PLOTS_DIR, FIGURES_DIR
from python_code.plotting.basic_plotter import BOTTOM_LIM, COLORS_DICT, MARKERS_DICT
from python_code.trainers.CVA.cva_trainer import CVATrainer
from python_code.trainers.GatedWCVAE.gated_wcvae_trainer import GatedWCVAETrainer
from python_code.utils.python_utils import load_pkl, save_pkl
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import numpy as np
import os

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

BLUE = '#4285F4'
D_MARKER = 'D'

POLY_NUM = 3
alphas = [4, 8, 16, 32]
info_length = 13
crc_length = 16
run_over = False
SNR_start = -2
SNR_end = 2
num_SNR = 5
snr_range = np.linspace(SNR_start, SNR_end, num=num_SNR)

plt.figure()

method_name = '3-rep CVA'
plots_path = os.path.join(PLOTS_DIR, method_name + '.pkl')
if os.path.isfile(plots_path) and not run_over:
    print("Loading plots")
    to_plot_curves = load_pkl(plots_path)
else:
    print("calculating fresh")
    cva_trainer = CVATrainer(run_name=None,
                             load_from_checkpoint=False,
                             val_SNR_start=SNR_start,
                             val_SNR_end=SNR_end,
                             val_num_SNR=num_SNR,
                             info_length=info_length,
                             code_length=POLY_NUM * (info_length + crc_length))
    to_plot_curves = cva_trainer.evaluate()
    save_pkl(plots_path, to_plot_curves)
plt.plot(snr_range, to_plot_curves[1], label=method_name,
         color=COLORS_DICT[re.sub("\d", "x", method_name)], linewidth=2.2,
         marker=MARKERS_DICT[re.sub("\d", "x", method_name)], linestyle='-.', markersize=12)

linestyles = ['-', '--', '-.', ':']
for alpha, linestyle in zip(alphas, linestyles):
    neural_state_run_name = 'gated_wcvae_{}'.format(info_length)

    plots_path = os.path.join(PLOTS_DIR, neural_state_run_name + '_' + str(alpha) + '.pkl')

    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        to_plot_curves = load_pkl(plots_path)
    else:
        print("calculating fresh")
        trainer = GatedWCVAETrainer(run_name=neural_state_run_name,
                                    load_from_checkpoint=True,
                                    val_SNR_start=SNR_start,
                                    val_SNR_end=SNR_end,
                                    val_num_SNR=num_SNR,
                                    info_length=info_length,
                                    crc_length=crc_length,
                                    decoders_in_ensemble=alpha,
                                    code_length=POLY_NUM * (info_length + crc_length))
        to_plot_curves = trainer.evaluate()
        save_pkl(plots_path, to_plot_curves)

    plt.plot(snr_range, to_plot_curves[1], label=f'Gated {alpha}-WCVAE',
             color=BLUE, linewidth=2.2,
             marker=D_MARKER, linestyle=linestyle, markersize=12)

LGVA_curve = [3e-2, 5.5e-3, 3e-4, 8e-6, 1e-6]
method_name = 'LGVA'
plt.plot(snr_range, LGVA_curve, label=method_name,
         color=COLORS_DICT[re.sub("\d", "x", method_name)], linewidth=2.2,
         marker=MARKERS_DICT[re.sub("\d", "x", method_name)], linestyle='-.', markersize=12)

plt.yscale('log')
plt.ylabel('FER')
plt.xlabel('$E_b/N_0$ [dB]')
plt.grid(which='both', ls='--')
plt.xlim([snr_range[0] - 0.1, snr_range[-1] + 0.1])
plt.legend(loc='lower left', prop={'size': 15}, handlelength=3.4)
plt.ylim(bottom=BOTTOM_LIM, top=1)

current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}_{info_length}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))

plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'FER.png'), bbox_inches='tight')

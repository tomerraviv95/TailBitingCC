import math

import torch

from python_code.utils.python_utils import load_pkl, save_pkl
import datetime
import os
import matplotlib as mpl
from dir_definitions import FIGURES_DIR, PLOTS_DIR
from python_code.trainers.CVA.cva_trainer import CVATrainer
from python_code.trainers.GatedWCVAE.gated_wcvae_trainer import GatedWCVAETrainer
from python_code.utils.metrics import calculate_error_rates
import matplotlib.pyplot as plt
import numpy as np

POLY_NUM = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['font.size'] = 19
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [8.2, 6.45]
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 17
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {
    'Gated CVAE': '#0F9D58',  # green
    'Gated WCVAE': 'cyan',
}
MARKERS_DICT = {'Gated CVAE': 'o',
                'Gated WCVAE': '*'}  # blue

run_over = False
gating_trainer = CVATrainer(log_flag=False)
gating_trainer.decoder.clipping_val = 0
channel_dataset = gating_trainer.channel_dataset['val']
neural_state_run_name = f'gated_wcvae_{gating_trainer.info_length}'
trainers = [GatedWCVAETrainer(run_name=None,
                              load_from_checkpoint=False,
                              info_length=gating_trainer.info_length,
                              crc_length=gating_trainer.crc_length,
                              code_length=POLY_NUM * (gating_trainer.info_length + gating_trainer.crc_length)),
            GatedWCVAETrainer(run_name=neural_state_run_name,
                              load_from_checkpoint=True,
                              info_length=gating_trainer.info_length,
                              crc_length=gating_trainer.crc_length,
                              code_length=POLY_NUM * (gating_trainer.info_length + gating_trainer.crc_length))]


def run_eval_per_state(trainer) -> np.array:
    total_fer = np.zeros([trainer.decoders_in_ensemble, trainer.n_states // trainer.decoders_in_ensemble])

    # pass through each decoder, one at a time
    for i in range(trainer.decoders_in_ensemble):
        states_cover = trainer.n_states // trainer.decoders_in_ensemble * np.array([i, i + 1])
        for j in range(states_cover[0], states_cover[1]):
            print(i, j)
            err_count = 0
            runs_num = 0
            while err_count < trainer.thresh_errors:
                current_state_cover = np.array([j, j])
                channel_dataset.set_states_cover(current_state_cover)

                # draw words only given by the decoder's state
                transmitted_words, received_words = iter(channel_dataset[:len(trainer.snr_range['val'])])
                transmitted_words = transmitted_words.to(device=device)
                received_words = received_words.to(device=device)

                # decode and calculate accuracy
                decoded_words = trainer.decoders_trainers[i].decoder(received_words, 'val')

                ber, fer, err_indices = calculate_error_rates(decoded_words, transmitted_words)
                err_count += err_indices.shape[0]
                total_fer[i, j % trainer.decoders_in_ensemble] += fer
                runs_num += 1.0

            total_fer[i, j % trainer.decoders_in_ensemble] /= runs_num

    return total_fer


fig, ax = plt.subplots(2, 4, gridspec_kw={'hspace': 0.2, 'wspace': 0})
SUBPLOT_LOC_TO_STATES = {(0, 0): (0, 8),
                         (0, 1): (8, 16),
                         (0, 2): (16, 24),
                         (0, 3): (24, 32),
                         (1, 0): (32, 40),
                         (1, 1): (40, 48),
                         (1, 2): (48, 56),
                         (1, 3): (56, 64)}
for trainer in trainers:
    name = '_'.join([trainer.get_name(), str(trainer.code_length), str(trainer.info_length), 'by', 'state'])
    plots_path = os.path.join(PLOTS_DIR, name + '.pkl')
    if os.path.isfile(plots_path) and not run_over:
        print("Loading plots")
        total_fer = load_pkl(plots_path)
    else:
        print("calculating fresh")
        total_fer = run_eval_per_state(trainer)
        save_pkl(plots_path, total_fer)

    states_v = np.arange(trainer.n_states)
    for loc, states_cover in SUBPLOT_LOC_TO_STATES.items():
        current_ax = ax[loc[0], loc[1]]
        current_ax.plot(states_v[states_cover[0]:states_cover[1]], total_fer.flatten()[states_cover[0]:states_cover[1]],
                        label=trainer.get_name().split(' ')[1][:-1] + ' Decoder',
                        color=COLORS_DICT[trainer.get_name()], linewidth=2.2,
                        marker=MARKERS_DICT[trainer.get_name()], linestyle='-.', markersize=10)
        current_ax.yaxis.set_ticks([])
        current_ax.yaxis.set_visible(False)
        current_ax.set_yscale('log')
        current_ax.set_xlim([states_cover[0], states_cover[1] - 1])
        current_ax.set_xticks(range(states_cover[0], states_cover[1]))
        if loc == (0, 0) or loc == (1, 0):
            xticks_labels = [str(int(states_cover[0])), '', '', '',
                             str(math.ceil((states_cover[0] + states_cover[1]) / 2)), '', '',
                             str(int(states_cover[1] - 1))]
            current_ax.set_xticklabels(xticks_labels)
        else:
            xticks_labels = ['', '', '', '', str(math.ceil((states_cover[0] + states_cover[1]) / 2)), '', '',
                             str(int(states_cover[1] - 1))]
            current_ax.set_xticklabels(xticks_labels)
fig.text(0.5, 0.02, 'States', ha='center')
fig.text(0.06, 0.5, 'FER', va='center', rotation='vertical')

current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))

plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'FER.png'), bbox_inches='tight')

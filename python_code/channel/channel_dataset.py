import collections
import concurrent.futures
import numpy as np
import torch
from numpy.random import mtrand
from torch.utils.data import Dataset
from typing import Tuple, List
from python_code.channel.channel import AWGNChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.tail_biting_utils import calculate_starting_state_for_tbcc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORKERS = 16


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation to draw minibatches of channel words.
    """

    def __init__(self, code_length: int,
                 info_length: int,
                 size_per_snr: int,
                 snr_range: List[int],
                 random: mtrand.RandomState,
                 word_rand_gen: mtrand.RandomState,
                 code_gm_inner: torch.Tensor,
                 code_gm_outer: torch.Tensor,
                 det_length: int,
                 phase: str,
                 training_words_factor: int,
                 n_states: int):

        self.code_length = code_length
        self.info_length = info_length

        self.random = random if random else np.random.RandomState()
        self.word_rand_gen = word_rand_gen if word_rand_gen else np.random.RandomState()
        self.modulation = BPSKModulator
        self.channel = AWGNChannel

        self.size_per_snr = size_per_snr
        self.snr_range = snr_range

        self.det_length = det_length
        self.training_words_factor = training_words_factor if phase is 'train' else 1

        self.n_states = n_states
        self.states_cover = None

        self.encoding_outer = lambda u: (torch.mm(u, code_gm_outer) % 2)
        self.encoding_inner = lambda c: (torch.mm(c, code_gm_inner) % 2)

    def set_states_cover(self, states_cover: np.ndarray):
        self.states_cover = states_cover

    def filter_by_states(self, u_det: torch.Tensor) -> torch.Tensor:
        """
        Filters words by states, leaving only states in the range of the states cover
        :param u_det: ground truth words
        :return: bool indices of words satisfying the states range
        """
        start_states = calculate_starting_state_for_tbcc(self.n_states, u_det)
        is_above_minimal_state = (self.states_cover[0] <= start_states)
        is_below_maximal_state = (start_states <= self.states_cover[1])
        total_ind = is_above_minimal_state * is_below_maximal_state
        return total_ind

    def get_snr_data(self, snr: int, database: list):
        if database is None:
            database = []
        u_full = torch.empty((0, self.info_length)).to(device=device)
        u_det_full = torch.empty((0, self.det_length)).to(device=device)
        c_full = torch.empty((0, self.code_length)).to(device=device)
        y_full = torch.empty((0, self.code_length)).to(device=device)
        # accumulate words until reaches desired number
        while y_full.shape[0] < self.size_per_snr:
            # random word generation
            # generate word
            u_array = self.word_rand_gen.randint(0, 2,
                                                 size=(
                                                     self.training_words_factor * self.size_per_snr, self.info_length))
            u = torch.Tensor(u_array).to(device=device)
            # outer encoding - errors detection code
            u_det = self.encoding_outer(u)
            # inner encoding - errors correction code
            c = self.encoding_inner(u_det)
            # modulation
            x = self.modulation.modulate(c)
            # transmit through noisy channel
            y = self.channel.transmit(x=x, SNR=snr, random=self.random).float()

            if self.states_cover is not None:
                total_ind = self.filter_by_states(u_det)
                u, u_det = u[total_ind], u_det[total_ind]
                c, y = c[total_ind], y[total_ind]

            # accumulate
            u_full = torch.cat((u_full, u), dim=0)
            u_det_full = torch.cat((u_det_full, u_det), dim=0)
            c_full = torch.cat((c_full, c), dim=0)
            y_full = torch.cat((y_full, y), dim=0)

        database.append((u_full[:self.size_per_snr], u_det_full[:self.size_per_snr],
                         c_full[:self.size_per_snr], y_full[:self.size_per_snr]))

    def __getitem__(self, snr_ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.snr_range, collections.Iterable):
            self.snr_range = [self.snr_range]
        if not isinstance(snr_ind, slice):
            snr_ind = [snr_ind]
        database = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            {executor.submit(self.get_snr_data, snr, database) for snr in self.snr_range[snr_ind]}
        u, u_det, c, y = (torch.cat(tensors) for tensors in zip(*database))
        u, u_det, c, y = u.to(device=device), u_det.to(device=device), c.to(device=device), y.to(device=device)
        return u_det, y

    def __len__(self):
        return self.size_per_snr * len(self.snr_range)

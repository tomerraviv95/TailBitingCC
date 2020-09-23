from python_code.utils.tail_biting_utils import create_transition_table
from python_code.utils.link import Link
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CVADecoder(nn.Module):
    """
    This implements the CVA decoder by unfolding into a neural network
    """

    def __init__(self, det_length: int,
                 replications: int,
                 n_states: int,
                 clipping_val: int,
                 code_length: int,
                 code_gm: torch.Tensor):

        super(CVADecoder, self).__init__()
        self.start_state = None
        self.initial_in_prob = None
        self.clipping_val = clipping_val
        self.det_length = det_length
        self.replications = replications
        self.check_replications()
        self.rate_inverse = code_length // self.det_length
        self.n_states = n_states
        self.code_gm = code_gm.cpu().numpy()

        # initialize states transition table
        self.transition_table = torch.Tensor(create_transition_table(self.n_states)).to(device)

        # initialize all stages of the cva decoders
        self.init_layers()

    def check_replications(self):
        if self.replications % 2 != 1:
            raise ValueError("Replications must be odd!!!")

    def init_layers(self):
        self.basic_layer = Link(n_states=self.n_states,
                                transition_table=self.transition_table,
                                rate_inverse=self.rate_inverse,
                                code_gm=self.code_gm).to(device)

    def set_start_state(self, start_states):
        self.start_state = start_states

    def forward(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The circular Viterbi algorithm
        :param x: input llrs (batch)
        :param phase: 'val' or 'train'
        :return: batch of decoded binary words
        """
        self.run(x)
        return self.traceback(phase)

    def run(self, x: torch.Tensor):
        """
        The forward pass of the circular Viterbi algorithm
        :param x: input llrs (batch)
        """
        batch_size = x.size(0)
        x = x.repeat(1, self.replications)

        if self.start_state is None:
            self.start_state = 0

        # set initial probabilities
        self.initial_in_prob = torch.zeros((batch_size, self.n_states)).to(device)
        self.initial_in_prob[torch.arange(batch_size).long(), self.start_state * torch.ones(batch_size).to(
            device).long()] = 0

        previous_states = torch.zeros([batch_size, self.n_states, self.det_length * self.replications]).to(device)
        out_prob_mat = torch.zeros([batch_size, self.n_states, self.det_length * self.replications]).to(device)
        in_prob = self.initial_in_prob.clone()

        for i in range(self.replications * self.det_length):
            out_prob, inds = self.basic_layer(in_prob, x[:, self.rate_inverse * i:self.rate_inverse * (i + 1)])
            # update the previous state (each index corresponds to the state out of the total n_states)
            previous_states[:, :, i] = self.transition_table[torch.arange(self.n_states).repeat(batch_size, 1), inds]
            out_prob_mat[:, :, i] = out_prob
            # update in-probabilities for next layer, clipping above and below thresholds
            in_prob = out_prob

        self.batch_size = batch_size
        self.previous_states = previous_states

    def traceback(self, phase: str) -> torch.Tensor:
        """
        Trace-back of the CVA
        :return: binary decoded codewords
        """
        if phase == 'val':
            # trace back unit
            most_likely_state = self.start_state
            ml_path_bits = torch.zeros([self.batch_size, self.det_length * self.replications]).to(device)

            # traceback - loop on all stages, from last to first, saving the most likely path
            for i in range(self.det_length * self.replications - 1, -1, -1):
                ml_path_bits[:, i] = (most_likely_state + 1) % 2
                most_likely_state = self.previous_states[torch.arange(self.batch_size), most_likely_state, i].long()

            decoded_words = ml_path_bits[:,
                            int(self.det_length * np.floor(self.replications / 2)):int(
                                self.det_length * np.ceil(self.replications / 2))]

            return decoded_words
        else:
            raise NotImplementedError("No implemented training for this decoder!!!")

    def get_iterations(self) -> float:
        return self.replications

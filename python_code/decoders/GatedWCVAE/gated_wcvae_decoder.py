from python_code.decoders.CVA.cva_decoder import CVADecoder
from typing import Dict, Tuple
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INF_CRC_VAL = 100


class GatedWCVAEDecoder(nn.Module):
    """
    This implements the CVA decoder by unfolding into a neural network
    """

    def __init__(self, det_length: int, code_h_outer: torch.Tensor, decoders_in_ensemble: int, n_states: int,
                 decoders_trainers: Dict, gating_decoder: CVADecoder, code_gm_inner: torch.Tensor):
        super(GatedWCVAEDecoder, self).__init__()
        self.det_length = det_length
        self.code_h_outer = code_h_outer
        self.decoders_in_ensemble = decoders_in_ensemble
        self.decoders_trainers = decoders_trainers
        self.gating_decoder = gating_decoder
        self.code_gm_inner = code_gm_inner
        self.n_states = n_states

    def forward(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The weighted CVA ensemble
        :param x: input llrs (batch)
        :param phase: 'val' or 'train'
        :return: batch of decoded binary words
        """
        decoded_words, non_zero_crc_ind, crc_sums = self.gating_wrapper(x)

        # if all words are satisfied, return
        if torch.sum(non_zero_crc_ind) == 0:
            return decoded_words

        x_hard = x[non_zero_crc_ind]
        batch_size = x_hard.size(0)
        total_decoded_words = torch.zeros([batch_size,
                                           self.det_length,
                                           self.decoders_in_ensemble]).to(device)
        # decode minimum values only, multiple experts runs if the minimum appears several times
        to_decode = (torch.min(crc_sums[non_zero_crc_ind], dim=1)[0].reshape(-1, 1) == crc_sums[
            non_zero_crc_ind]).bool()

        crc_sums = torch.zeros([batch_size, self.decoders_in_ensemble]).to(device)
        crc_sums[~to_decode] = INF_CRC_VAL

        for i in range(self.decoders_in_ensemble):
            # run through the appropriate weighted decoder
            if torch.sum(to_decode[:, i]) > 0:
                current_decoded_words = self.decoders_trainers[i].decoder(x_hard[to_decode[:, i]], 'val')
                total_decoded_words[to_decode[:, i], :, i] = current_decoded_words.clone()

                # calculate crc for decoded word
                crc_sums[to_decode[:, i], i] = torch.sum(torch.matmul(self.code_h_outer, current_decoded_words.T) % 2, dim=0)

        # choose minimum crc decoded words
        ind = torch.argmin(crc_sums, dim=1).reshape(-1)
        decoded_words[non_zero_crc_ind] = total_decoded_words[torch.arange(batch_size), :, ind]

        # keep parameters for calculation of average iterations, taking into account only active runs
        self.unsat_words_per_dec = to_decode.sum(dim=0).float()
        self.non_zero_crc_ind = non_zero_crc_ind

        return decoded_words

    def gating_wrapper(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the gating and multiple trace-backs, each trace-back from a different state
        :param x: input llrs (batch)
        :return: decoded codewords, indices of non zero crc locations and the crc checks sums
        """
        # forward pass
        self.gating_decoder.run(x)

        # initialize tensors
        batch_size = x.shape[0]
        total_decoded_words = torch.zeros([batch_size,
                                           self.det_length,
                                           self.decoders_in_ensemble]).to(device)
        crc_sums = torch.zeros([batch_size, self.decoders_in_ensemble]).to(device).float()

        # run for each start state, one start state for each expert in the WCVAE
        for i in range(self.decoders_in_ensemble):
            # run through the appropriate weighted decoder
            start_state = int(self.n_states // self.decoders_in_ensemble * (i + 0.5))
            self.gating_decoder.set_start_state(start_state)

            # trace-back
            decoded_words = self.gating_decoder.traceback('val')
            total_decoded_words[:, :, i] = decoded_words.clone()

            # calculate crc for decoded word
            hard_crc_values = (torch.matmul(self.code_h_outer, decoded_words.T) % 2).T
            crc_sums[:, i] = torch.sum(hard_crc_values, dim=1).float()

        gating_ind = torch.argmin(crc_sums, dim=1).reshape(-1)
        final_crc_checks = crc_sums[torch.arange(batch_size), gating_ind]
        decoded_words = total_decoded_words[torch.arange(x.shape[0]), :, gating_ind]
        non_zero_crc_ind = (final_crc_checks != 0)
        return decoded_words, non_zero_crc_ind, crc_sums

    def get_iterations(self) -> float:
        gating_iterations = self.gating_decoder.replications
        decoders_iterations = torch.Tensor(
            [self.decoders_trainers[i].decoder.get_iterations() for i in range(self.decoders_in_ensemble)]).to(device)
        ensemble_iterations = (torch.matmul(decoders_iterations,
                                            self.unsat_words_per_dec) / self.non_zero_crc_ind.shape[0]).item()
        return gating_iterations + ensemble_iterations

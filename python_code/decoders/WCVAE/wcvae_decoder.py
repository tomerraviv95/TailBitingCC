from typing import Dict
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WCVAEDecoder(nn.Module):
    """
    This implements the WCVAE Decoder
    """

    def __init__(self, det_length: int, code_h_outer: torch.Tensor, decoders_in_ensemble: int, n_states: int,
                 decoders_trainers: Dict, code_gm_inner: torch.Tensor):
        super(WCVAEDecoder, self).__init__()
        self.det_length = det_length
        self.code_h_outer = code_h_outer
        self.decoders_in_ensemble = decoders_in_ensemble
        self.decoders_trainers = decoders_trainers
        self.code_gm_inner = code_gm_inner
        self.n_states = n_states

    def forward(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The WCVA ensemble
        :param x: input llrs (batch)
        :param phase: 'val' or 'train'
        :return: batch of decoded binary words
        """

        x_hard = x
        batch_size = x_hard.size(0)
        total_decoded_words = torch.zeros([batch_size,
                                           self.det_length,
                                           self.decoders_in_ensemble]).to(device)
        to_decode = torch.ones([batch_size, self.decoders_in_ensemble]).bool()
        crc_sums = torch.zeros([batch_size, self.decoders_in_ensemble]).to(device)

        # run through each decoder
        for i in range(self.decoders_in_ensemble):
            # decode with the appropriate weighted decoder
            current_words_to_decode = x_hard[to_decode[:, i]]
            decoded_words = self.decoders_trainers[i].decoder(current_words_to_decode, 'val')
            total_decoded_words[to_decode[:, i], :, i] = decoded_words.clone()
            # calculate crc sum for decoded word
            crc_sums[to_decode[:, i], i] = torch.sum(torch.matmul(self.code_h_outer, decoded_words.T) % 2, dim=0)

        ind = torch.argmin(crc_sums, dim=1).reshape(-1)
        chosen_words = total_decoded_words[torch.arange(batch_size), :, ind]

        return chosen_words

    def get_iterations(self) -> float:
        decoders_iterations = torch.Tensor(
            [self.decoders_trainers[i].decoder.get_iterations() for i in range(self.decoders_in_ensemble)]).to(device)
        ensemble_iterations = torch.sum(decoders_iterations).item()
        return ensemble_iterations

import torch
from numpy.random import mtrand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AWGNChannel:
    @staticmethod
    def transmit(x: torch.Tensor, SNR: int, random: mtrand.RandomState):
        """
        The AWGN Channel
        :param x: to transmit codeword
        :param SNR: signal-to-noise value
        :param random: random words generator
        :param use_llr: whether llr values or magnitude
        :return: received word
        """
        [row, col] = x.shape

        sigma = 10 ** (-SNR / 20)

        y = x + sigma * torch.Tensor(random.normal(0.0, 1.0, (row, col))).to(device)

        return 2 * y / (sigma ** 2)

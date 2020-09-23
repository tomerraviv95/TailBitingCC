import torch
import torch.nn as nn
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Link(nn.Module):
    def __init__(self, n_states: int,
                 transition_table: torch.Tensor,
                 rate_inverse: float,
                 code_gm: np.ndarray):
        self.n_states = n_states
        self.transition_table = transition_table
        self.rate_inverse = rate_inverse
        self.code_gm = code_gm
        super().__init__()

        # create matrices
        self.create_states_to_edges_matrix()
        self.create_llrs_to_edges_matrix()

    def create_states_to_edges_matrix(self):
        self.states_to_edges_mask = torch.cat([self.transition_table.reshape(1, -1) == i for i in range(self.n_states)],
                                              dim=0).float().to(device)
        self.states_to_edges = nn.Parameter(self.states_to_edges_mask, requires_grad=False).to(device)

    def create_llrs_to_edges_matrix(self):
        self.create_all_llrs_combinations()
        self.create_llrs_combinations_to_edges()
        llrs_to_edges_mask = torch.mm(self.all_llrs_combinations, self.llrs_combinations_to_edges).to(device)
        self.llrs_to_edges = nn.Parameter(llrs_to_edges_mask, requires_grad=False).to(device)

    def create_all_llrs_combinations(self):
        """
        A matrix of size (1/R) X 2**(1/R).
        Enumerates all possible combinations of the (1/R) llr values at the encoder's output.
        """
        binary_combinations = np.array(list(itertools.product(range(2), repeat=self.rate_inverse)))
        bpsk_mapped = (-1) ** binary_combinations
        self.all_llrs_combinations_mat = np.fliplr(np.flipud(bpsk_mapped)).copy()
        self.all_llrs_combinations = torch.Tensor(self.all_llrs_combinations_mat).T

    def create_llrs_combinations_to_edges(self):
        """
        A matrix of size 2**(1/R) X 2*n_states.
        Enumerates for each column all relevant edges. There are 2*n_states edges. cell [i,j] is active if
        and only if edge i has combination j of llr at it's output.
        """

        generator_polys = self.code_gm[0, :self.rate_inverse * (int(np.log2(self.n_states)) + 1)]
        generator_polys = generator_polys.reshape(int(np.log2(self.n_states)) + 1, -1).T
        generator_polys = np.fliplr(generator_polys)
        states_binary_combinations = np.array(
            list(itertools.product(range(2), repeat=int(np.log2(self.n_states))))).repeat(2, axis=0)
        input_bits = np.tile(np.array([1, 0]), self.n_states).reshape(-1, 1)

        binary_combinations = np.concatenate([input_bits, states_binary_combinations], axis=1)
        bits_outputs_on_edges = np.matmul(binary_combinations, generator_polys.T) % 2
        llr_outputs_on_edges = (-1) ** bits_outputs_on_edges
        llrs_combinations_to_edges_mat = np.zeros([2 ** self.rate_inverse, 2 * self.n_states])

        for row_ind in range(llrs_combinations_to_edges_mat.shape[0]):
            llrs_combinations_to_edges_mat[row_ind] = np.equal(llr_outputs_on_edges,
                                                               self.all_llrs_combinations_mat[row_ind]).all(1)

        self.llrs_combinations_to_edges = torch.Tensor(llrs_combinations_to_edges_mat)

    def compare_select(self, x: torch.Tensor) -> [torch.Tensor, torch.LongTensor]:
        """
        The compare-select operation return the maximum probabilities and the edges' indices of the chosen
        maximal values.
        :param x: LLRs matrix of size [batch_size,2*n_states] - two following indices in each row belong to two
        competing edges that enter the same state
        :return: the maximal llrs (from every pair), and the absolute edges' indices
        """
        reshaped_x = x.reshape(-1, self.n_states, 2)
        max_values, absolute_max_ind = torch.max(reshaped_x, 2)
        return max_values, absolute_max_ind

    def forward(self, in_prob: torch.Tensor, llrs: torch.Tensor) -> [torch.Tensor, torch.LongTensor]:
        """
        Viterbi ACS block
        :param in_prob: last stage probabilities, [batch_size,n_states]
        :param llrs: edge probabilities, [batch_size,rate_inverse]
        :return: current stage probabilities, [batch_size,n_states]
        """
        A = torch.mm(in_prob, self.states_to_edges * self.states_to_edges_mask)
        B = torch.mm(llrs, self.llrs_to_edges)
        return self.compare_select(A + B)

    def apply_grad(self):
        self.llrs_to_edges.requires_grad = True
        self.states_to_edges.requires_grad = True

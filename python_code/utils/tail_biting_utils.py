import numpy as np
import itertools
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UINT8_CONSTANT = 8


def calculate_starting_state_for_tbcc(n_states: int, u_det: torch.Tensor):
    """
    calculates the starting state for the give words (u_det is information + crc word)
    take last bits, and pass through code's trellis
    :param u_det: size [batch_size,info_length+crc_length]
    :return: vector of length batch_size with values in the range of 0,1,...,n_states-1
    """
    tail_biting_bits = int(math.log2(n_states))
    last_bits_of_u_det = u_det[:, -tail_biting_bits:]
    start_states = torch.zeros([u_det.size(0), tail_biting_bits + 1]).long().to(device)
    for i in range(tail_biting_bits):
        start_states[:, i + 1] = map_bit_and_state_to_next_state(n_states,
                                                                 last_bits_of_u_det[:, i],
                                                                 start_states[:, i])
    return start_states[:, -1]


def map_bit_and_state_to_next_state(n_states: int, bit: torch.Tensor, state: torch.Tensor):
    """
    Based on the current srs and bits arrays. For example:
    states 0 and 32 with input bit 1 are mapped to state 0
    states 0 and 32 with input bit 0 are mapped to state 1
    states 1 and 33 with input bit 1 are mapped to state 2
    states 1 and 33 with input bit 0 are mapped to state 3
    and so on...
    """
    next_state = (2 * state + (1 - bit.long())) % n_states
    return next_state.long()


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    next state of state i and input bit b is the state in cell [i,b]
    """
    all_states = list(itertools.product([0, 1], repeat=int(np.log2(n_states))))
    repeated_all_states = np.repeat(np.array(all_states), 2, axis=0)
    # create new states by inserting 0 or 1 for each state
    inserted_bits = np.tile(np.array([0, 1]), n_states).reshape(-1, 1)
    mapped_states = np.concatenate(
        [np.zeros([2 * n_states, UINT8_CONSTANT - repeated_all_states.shape[1]]), inserted_bits,
         repeated_all_states[:, :-1]],
        axis=1)
    numbered_mapped_states = np.packbits(mapped_states.astype(np.uint8), axis=1)
    transition_table = numbered_mapped_states.reshape(n_states, 2)
    return transition_table

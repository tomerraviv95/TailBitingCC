import torch
import numpy as np
import os
from dir_definitions import WEIGHTS_DIR
from python_code.decoders.WCVAE.wcvae_decoder import WCVAEDecoder
from python_code.trainers.WCVA.wcva_trainer import WCVATrainer
from python_code.trainers.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WCVAETrainer(Trainer):
    """
    Trainer for the WCVAE model.
    """

    def __init__(self, config_path=None, **kwargs):
        self.replications = None
        self.decoders_in_ensemble = None
        super().__init__(config_path, **kwargs)

    def __name__(self):
        if self.start_minibatch > 0:
            alg = 'WCVA'
        else:
            alg = 'CVA'
        return f'{alg}E'

    def load_decoder(self):
        """
        Loads the WCVA decoders
        """

        self.initialize_decoders()
        self.decoder = WCVAEDecoder(det_length=self.det_length,
                                    code_h_outer=self.code_h_outer,
                                    decoders_in_ensemble=self.decoders_in_ensemble,
                                    n_states=self.n_states,
                                    decoders_trainers=self.decoders_trainers,
                                    code_gm_inner=self.code_gm_inner)

    def initialize_decoders(self):
        """
         Initializes all WCVA trainers
         """
        self.decoders_trainers = {}
        for i in range(self.decoders_in_ensemble):
            self.decoders_trainers[i] = WCVATrainer(info_length=self.info_length,
                                                    det_length=self.det_length,
                                                    code_length=self.code_length)

            # set the relevant states cover
            states_cover = self.n_states // self.decoders_in_ensemble * np.array([i, i + 1])
            # apply permanent filtering to the dataloader - only for train phase
            self.decoders_trainers[i].channel_dataset['train'].set_states_cover(states_cover)
            # set appropriate starting state for the decoder
            self.decoders_trainers[i].decoder.set_start_state(int((states_cover[0] + states_cover[1]) / 2))

            # fix the run and output dir names
            self.decoders_trainers[i].run_name = f'{self.run_name}_{self.decoders_in_ensemble}_{i + 1}'
            self.decoders_trainers[i].weights_dir = os.path.join(WEIGHTS_DIR, self.decoders_trainers[i].run_name)
            if not os.path.isdir(self.decoders_trainers[i].weights_dir):
                os.makedirs(self.decoders_trainers[i].weights_dir)

            # load checkpoint
            self.load_checkpoint(i)

    def load_checkpoint(self, i: int):
        if self.load_from_checkpoint:
            self.decoders_trainers[i].load_last_checkpoint()

    def deep_learning_setup(self):
        """
        Implemented for an ensemble - see how OO makes life simple
        """
        for j in range(self.decoders_in_ensemble):
            self.decoders_trainers[j].deep_learning_setup()

    def run_single_train_loop(self):
        """
        Implemented for an ensemble - see how OO makes life simple
        """
        loss = 0
        # pass through each decoder, one at a time
        for j in range(self.decoders_in_ensemble):
            loss += self.decoders_trainers[j].run_single_train_loop()
        return loss

    def save_checkpoint(self, current_loss: float, minibatch: int):
        """
        Implemented for an ensemble - see how OO makes life simple
        """
        for j in range(self.decoders_in_ensemble):
            self.decoders_trainers[j].save_checkpoint(current_loss, minibatch)


if __name__ == '__main__':
    dec = WCVAETrainer()
    dec.train()

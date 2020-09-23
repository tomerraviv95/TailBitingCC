from python_code.decoders.CVA.cva_decoder import CVADecoder
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CVATrainer(Trainer):
    """
    Trainer for the CVA model.
    """

    def __init__(self, config_path=None, **kwargs):
        self.replications = None
        super().__init__(config_path, **kwargs)

    def __name__(self):
        return f'{self.replications}-rep CVA'

    def load_decoder(self):
        """
        Loads the CVA decoder
        """

        self.decoder = CVADecoder(det_length=self.det_length,
                                  replications=self.replications,
                                  n_states=self.n_states,
                                  clipping_val=self.clipping_val,
                                  code_length=self.code_length,
                                  code_gm=self.code_gm_inner)

    def train(self):
        raise NotImplementedError("No training implemented for this decoder!!!")


if __name__ == '__main__':
    dec = CVATrainer()
    dec.evaluate()

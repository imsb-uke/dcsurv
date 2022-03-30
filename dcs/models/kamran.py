from dcs.models.dcs_model import DcsModel


class Kamran(DcsModel):
    def __init__(self, **kwargs):

        kamran_defaults = {
            'output_grid_type': 'linear',
            'encoder_num_layers': 1,
            'encoder_nodes_per_layer': 64,
            'decoder_num_layers': 1,
            'decoder_bidirectional': False,
            'decoder_use_lstm_skip': False,
            'output_num_layers': 1,
            'interpolate_method': 'linear',
            'loss_kernel_include_censored': False,
            'loss_normalize': False,
        }

        for kwarg in kwargs:
            if kwarg in kamran_defaults:
                raise ValueError(
                    f"'{kwarg}' is fixed to '{kamran_defaults[kwarg]}' "
                    "and cannot be used.")

        kwargs.update(kamran_defaults)

        super().__init__(**kwargs)

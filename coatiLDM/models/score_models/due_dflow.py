import torch
from torch import nn
from due.dkl import DKL, GP
from due.fc_resnet import FCResNet
import smart_open
import pickle


def load_inference_due_for_dflow(
    load_as,
    input_dim=512,
    n_inducing_points=60,
    depth=4,
    remove_spectral_norm=False,
    passed_state_dict=None,
):
    """just loads model for inference. Doesn't require associated dataset. can remove norm if desired

    Args:
        load_as (str): model path
        input_dim (int): input dimensions.
        n_inducing_points(int, optional): need this to load correctly, uses default from basic_due
        depth (int, optional): default shared with basic_due, but can also be modified
        remove_spectral_norm (bool, optional): remove spectral norm if taking gradients. Defaults to False.
    """

    features = 256
    num_outputs = 1
    spectral_normalization = True
    coeff = 0.95
    n_power_iterations = 2
    dropout_rate = 0.00

    feature_extractor = FCResNet(
        input_dim=input_dim,
        features=features,
        depth=depth,
        spectral_normalization=spectral_normalization,
        coeff=coeff,
        n_power_iterations=n_power_iterations,
        dropout_rate=dropout_rate,
    )

    kernel = "RBF"
    # The following will be loaded just need right shapes
    initial_inducing_points = torch.zeros((n_inducing_points, features))
    initial_lengthscale = torch.tensor(0.5)

    gp = GP(
        num_outputs=num_outputs,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=kernel,
    )
    model = DKL(feature_extractor, gp)

    if passed_state_dict:
        read = passed_state_dict
    else:
        with smart_open.open(load_as, "rb") as f_in:
            read = torch.load(f_in)
        # read = torch.load(load_as)
    model.load_state_dict(read)

    if remove_spectral_norm:
        model.feature_extractor.first = torch.nn.utils.remove_spectral_norm(
            model.feature_extractor.first
        )

    model.eval()
    return model

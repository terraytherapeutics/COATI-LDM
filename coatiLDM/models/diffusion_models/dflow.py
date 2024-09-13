# load data
import torch
from torchdiffeq import odeint


def dflow(
    x_start,
    target_set,
    vec_field_net,
    target_net,
    learning_rate=1.0,
    decode_steps=100,
    opt_steps=5,
    device="cuda:0",
):
    """
    Performs the DFlow optimization.

    Args:
        x_start (torch.Tensor): Starting tensor for the optimization.
        target_value (float): The target value to aim for in the optimization.
        vec_field_net (nn.Module): Vector field network.
        target_net (nn.Module): Target network.
        learning_rate (float): Learning rate for the optimizer.
        decode_steps (int): Number of decoding steps.
        opt_steps (int): Number of optimization steps.
        device (str): Device to run the computation on.

    Returns:
        torch.Tensor: The optimized tensor.
    """

    # def wrapper(t, x):
    #    t = t * torch.ones(len(x), device=device)
    #    return vec_field_net(x, t)

    def closure():
        optimizer.zero_grad()
        x_1 = odeint(vec_field_net, x_0, t, method="midpoint")
        mse_loss = torch.pow(target_net(x_1).mean - target_tensor, 2).mean()
        mse_loss.backward(retain_graph=True)
        return mse_loss

    batch_size = x_start.shape[0]

    target_tensor = target_set.clone().to(device)
    # print(target_tensor.shape)

    t = torch.linspace(0, 1, steps=decode_steps, device=device)
    x_0 = x_start.clone().detach().requires_grad_(True)  # Ensure x_0 requires grad

    optimizer = torch.optim.LBFGS([x_0], lr=learning_rate, max_iter=5)

    for ii in range(opt_steps):
        optimizer.step(closure)
        # x_0 = x_0.detach().requires_grad_(True)

    x_1 = odeint(vec_field_net, x_0, t, method="midpoint")
    return x_1.detach()


def dflow_multi(
    x_start,
    target_sets,
    vec_field_net,
    target_nets,
    learning_rate=1.0,
    decode_steps=100,
    opt_steps=5,
    device="cuda:0",
):
    """
    Performs the DFlow optimization for multiple targets.

    Args:
        x_start (torch.Tensor): Starting tensor for the optimization.
        target_values (list of float): List of target values to aim for in the optimization.
        vec_field_net (nn.Module): Vector field network.
        target_nets (list of nn.Module): List of target networks.
        learning_rate (float): Learning rate for the optimizer.
        decode_steps (int): Number of decoding steps.
        opt_steps (int): Number of optimization steps.
        device (str): Device to run the computation on.

    Returns:
        torch.Tensor: The optimized tensor.
    """

    def closure():
        optimizer.zero_grad()
        x_1 = odeint(vec_field_net, x_0, t, method="midpoint")
        mse_losses = [
            torch.pow(net(x_1).mean - target_tensor, 2).mean()
            for net, target_tensor in zip(target_nets, target_tensors)
        ]
        total_loss = sum(mse_losses)
        total_loss.backward(retain_graph=True)
        return total_loss

    batch_size = x_start.shape[0]

    target_tensors = [val.clone().to(device) for val in target_sets]

    t = torch.linspace(0, 1, steps=decode_steps, device=device)
    x_0 = x_start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([x_0], lr=learning_rate, max_iter=5)

    for ii in range(opt_steps):
        optimizer.step(closure)

    x_1 = odeint(vec_field_net, x_0, t, method="midpoint")
    result = x_1.detach()
    del x_1, x_0, target_tensors, optimizer
    torch.cuda.empty_cache()  # Clear CUDA cache

    return result

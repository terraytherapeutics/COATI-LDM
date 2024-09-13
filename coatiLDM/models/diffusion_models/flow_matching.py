#
# Defines 3 types of flow matching and an adapter to create a
# conditional vector field
# shamelessly borrowed from this excellent notebook:
# https://colab.research.google.com/github/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb#scrollTo=l_NPXHeSNg8Y
#

import torch
import torch.nn as nn

from zuko.utils import odeint


class CondOTFlowMatching:
    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(
        self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional Flow"""
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, cs) -> torch.Tensor:
        """Compute loss"""
        # t ~ Unif([0, 1])
        t = (
            torch.rand(1, device=x_1.device)
            + torch.arange(len(x_1), device=x_1.device) / len(x_1)
        ) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x_0)
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t), cs)
        d_psi = x_1 - (1 - self.sig_min) * x_0
        return torch.mean((v_psi - d_psi) ** 2)


class OTFlowMatching:
    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(
        self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Conditional Flow"""
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        # t ~ Unif([0, 1])
        t = (
            torch.rand(1, device=x_1.device)
            + torch.arange(len(x_1), device=x_1.device) / len(x_1)
        ) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x_0)
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        d_psi = x_1 - (1 - self.sig_min) * x_0
        return torch.mean((v_psi - d_psi) ** 2)


class VEDiffusionFlowMatching:
    def __init__(self) -> None:
        super().__init__()
        self.sigma_min = 0.01
        self.sigma_max = 2.0
        self.eps = 1e-5

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_t(t) * torch.log(
            torch.tensor(self.sigma_max / self.sigma_min)
        )

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return -(self.dsigma_dt(1.0 - t) / self.sigma_t(1.0 - t)) * (x - x_1)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        # t ~ Unif([0, 1])
        t = (
            torch.rand(1, device=x_1.device)
            + torch.arange(len(x_1), device=x_1.device) / len(x_1)
        ) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = x_1 + self.sigma_t(1.0 - t) * torch.randn_like(x_1)
        return torch.mean((v_t(t[:, 0], x) - self.u_t(t, x, x_1)) ** 2)


class VPDiffusionFlowMatching:
    def __init__(self) -> None:
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.eps = 1e-5

    def T(self, s: torch.Tensor) -> torch.Tensor:
        return self.beta_min * s + 0.5 * (s**2) * (self.beta_max - self.beta_min)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.T(t))

    def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return self.alpha(1.0 - t) * x_1

    def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alpha(1.0 - t) ** 2)

    def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        num = torch.exp(-self.T(1.0 - t)) * x - torch.exp(-0.5 * self.T(1.0 - t)) * x_1
        denum = 1.0 - torch.exp(-self.T(1.0 - t))
        return -0.5 * self.beta(1.0 - t) * (num / denum)

    def loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        # t ~ Unif([0, 1])
        t = (
            torch.rand(1, device=x_1.device)
            + torch.arange(len(x_1), device=x_1.device) / len(x_1)
        ) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x|x_1)
        x = self.mu_t(t, x_1) + self.sigma_t(t, x_1) * torch.randn_like(x_1)
        return torch.mean((v_t(t[:, 0], x) - self.u_t(t, x, x_1)) ** 2)


class CondVF(nn.Module):
    """
    conditional vector field...
    """

    def __init__(self, net: nn.Module, n_steps: int = 100) -> None:
        super().__init__()
        self.net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args t: a zero-D tensor of time in [0,1]
        """
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def decode_t0_t1(self, x_0, t0, t1):
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1: torch.Tensor) -> torch.Tensor:
        return odeint(self.wrapper, x_1, 1.0, 0.0, self.parameters())

    def decode(self, x_0: torch.Tensor) -> torch.Tensor:
        return odeint(self.wrapper, x_0, 0.0, 1.0, self.parameters())


class CondCondVF(nn.Module):
    """
    conditional vector field...
    """

    def __init__(self, net: nn.Module, n_steps: int = 100) -> None:
        super().__init__()
        self.net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor, cs) -> torch.Tensor:
        return self.net(t, x, cs)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor, cs) -> torch.Tensor:
        """
        Args t: a zero-D tensor of time in [0,1]
        """
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x, cs)

    def decode_t0_t1(self, x_0, t0, t1, cs):
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_0, t0, t1, self.parameters()
        )

    def encode(self, x_1: torch.Tensor, cs) -> torch.Tensor:
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_1, 1.0, 0.0, self.parameters()
        )

    def decode(self, x_0: torch.Tensor, cs) -> torch.Tensor:
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_0, 0.0, 1.0, self.parameters()
        )


class ScoreNetCondVF(nn.Module):
    """
    conditional vector field...
    """

    def __init__(self, net: nn.Module, n_steps: int = 100) -> None:
        super().__init__()
        self.score_net = net

    def forward(self, t: torch.Tensor, x: torch.Tensor, cs) -> torch.Tensor:
        return self.score_net(x, t, cs)

    def wrapper(self, t: torch.Tensor, x: torch.Tensor, cs) -> torch.Tensor:
        """
        Args t: a zero-D tensor of time in [0,1]
        """
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x, cs)

    def decode_t0_t1(self, x_0, t0, t1, cs):
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_0, t0, t1, self.parameters()
        )

    def encode(self, x_1: torch.Tensor, cs) -> torch.Tensor:
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_1, 1.0, 0.0, self.parameters()
        )

    def decode(self, x_0: torch.Tensor, cs) -> torch.Tensor:
        return odeint(
            lambda T, X: self.wrapper(T, X, cs), x_0, 0.0, 1.0, self.parameters()
        )


_one_third = 1.0 / 3
_two_thirds = 2.0 / 3


def euler_step(func, x0, t0, dt=1.0 / 1000.0, f0=None):
    x1 = func(t0, x0)
    return x0 + dt * (x1)


def rk4_step(func, x0, t0, dt=1.0 / 1000.0, f0=None):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, x0)
    k2 = func(t0 + dt * _one_third, x0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, x0 + dt * (k2 - k1 * _one_third))
    k4 = func(t0 + dt, x0 + dt * (k1 - k2 + k3))
    return x0 + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


def ode_int_explicit(
    func, x0, t0=0.0, t1=1.0, nT=1000, differentiable=False, rule=euler_step
):
    """
    Solves
       d(X(t))/dt = func(t, x)
    """
    x = x0.clone()
    dt = (t1 - t0) / nT
    if differentiable:
        with torch.grad_enable():
            for i in range(nT):
                t = t0 + i * dt
                x = rule(func, x, t * torch.ones(x.shape[0], device=x.device), dt=dt)

    else:
        with torch.no_grad():
            for i in range(nT):
                t = t0 + i * dt
                x = rule(func, x, t * torch.ones(x.shape[0], device=x.device), dt=dt)
    return x


class OT_cond_flow_matching(nn.Module):
    """
    For use with allegro_vector_field
    """

    def __init__(self, score_net):
        super().__init__()
        self.score_net = score_net
        self.eps = 1e-5
        self.sig_min = 1e-5
        # self.sig_min = 0.001

    def forward(self, x1, x0, cond=None):
        """
        Returns the loss of this type of flow matching.
        Requires the prior x0 samples (same shape as x1)
        """
        t = (
            torch.rand(1, device=x1.device)
            + torch.arange(x1.shape[0], device=x1.device) / x1.shape[0]
        ) % (1 - self.eps)
        psi_t = (
            t.unsqueeze(-1) * x1 + (1.0 - (1.0 - self.sig_min) * t.unsqueeze(-1)) * x0
        )
        dpsi_dt = x1 - (1.0 - self.sig_min) * x0
        return torch.pow((self.score_net(psi_t, t, cond=cond) - dpsi_dt), 2.0).mean()

    def sample(
        self,
        x0,
        t0=0.0,
        t1=1.0,
        nT=1000,
        rule=euler_step,
        differentiable=False,
        cond=None,
    ):
        return ode_int_explicit(
            lambda t, x: self.score_net(x, t, cond=cond),
            x0,
            t0=t0,
            t1=t1,
            nT=nT,
            rule=rule,
            differentiable=differentiable,
        )

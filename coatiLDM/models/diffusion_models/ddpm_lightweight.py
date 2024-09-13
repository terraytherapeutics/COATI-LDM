import torch


class DDPMScoreNetTrainer(torch.nn.Module):
    def __init__(self, score_net):
        super().__init__()
        self.score_net = score_net
        self.scheduler = score_net.scheduler

    def forward(self, x, cond=None, loss_weight=None):
        batch_size = x.shape[0]
        device = next(self.score_net.parameters()).device
        T = torch.randint(
            low=0, high=self.scheduler.timesteps, size=(batch_size,), device=device
        )
        noise = torch.randn((batch_size, self.score_net.x_dim), device=device)
        noisy_samples = (
            self.scheduler.bar_alpha(T).sqrt() * x
            + (1.0 - self.scheduler.bar_alpha(T)).sqrt() * noise
        )
        extracted_noise = self.score_net(noisy_samples, t=T.float(), cond=cond)
        pre_weight = torch.pow(noise - extracted_noise, 2.0).mean(-1)
        if not loss_weight is None:
            assert loss_weight.shape[0] == batch_size
            assert loss_weight.dim() == 1
            return (pre_weight * loss_weight).mean()
        return pre_weight.mean()

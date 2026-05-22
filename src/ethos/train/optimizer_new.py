import inspect

import torch


def _newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize G via Newton-Schulz iteration (quintic polynomial, 5 steps)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz — for 2D hidden-layer weights only.

    Applies Newton-Schulz orthogonalization to the Nesterov momentum buffer,
    achieving ~2x compute efficiency vs AdamW at GPT-2 scale.
    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                update = _newtonschulz5(update, steps=ns_steps)
                update.mul_(max(p.size(0), p.size(1)) ** 0.5)
                p.add_(update, alpha=-lr)


class CombinedOptimizer:
    """Wraps a (Muon, AdamW) pair behind a single optimizer interface.

    Supports proportional LR scheduling via scale_lr(ratio): each param group
    scales relative to its own base_lr, so Muon and AdamW can have different
    peak LRs while following the same cosine decay curve.
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw
        for pg in self.param_groups:
            pg["base_lr"] = pg["lr"]

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def scale_lr(self, ratio: float):
        for pg in self.param_groups:
            pg["lr"] = pg["base_lr"] * ratio

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = False):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def configure_optimizers_muon(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple,
    device_type: str,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
) -> CombinedOptimizer:
    """Return a CombinedOptimizer(Muon, AdamW) for ModernGPTModel.

    Muon:  all 2D hidden-layer weight matrices (attention + FFN projections)
    AdamW: token embedding (weight-decayed) + RMSNorm scales (no decay)
    """
    emb_ids = {id(model.transformer.wte.weight)}

    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1:
            adamw_nodecay.append(param)
        elif id(param) in emb_ids:
            adamw_decay.append(param)
        else:
            muon_params.append(param)

    muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum)

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device_type
    extra_args = dict(fused=True) if use_fused else dict()
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_decay, "weight_decay": weight_decay},
            {"params": adamw_nodecay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )

    return CombinedOptimizer(muon, adamw)

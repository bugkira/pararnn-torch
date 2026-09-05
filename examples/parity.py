"""Running parity (Z₂ tagging): stacked ParaSLSTM vs sequential vs SSM.

Merrill et al. 2024 §5: label at t is the prefix product on Z₂ (XOR).
T=16 / 2000 steps; eval also at T=32.

Usage:
    python parity.py
"""

import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pararnn import NewtonConfig, ParaRNN, ParaSLSTM
from pararnn.solvers.scan import scan_diag

VOCAB = 2
SEQ_LEN, EVAL_SEQ_LEN = 16, 32
BATCH, D_H, NUM_LAYERS = 16, 32, 2
STEPS, WARMUP_FRAC = 2000, 0.1
NEWTON_ITERS, SCAN_BACKEND = 3, "auto"
LR, WEIGHT_DECAY = 1e-3, 1e-6
ADAM_BETAS = (0.9, 0.999)
SEED = 0
ARMS = ("newton", "eager", "ssm")


def sample_parity(
    batch: int, length: int, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Bits and running XOR labels (Merrill §5 tagging on Z₂)."""
    bits = torch.randint(0, 2, (batch, length), generator=generator)
    labels = bits.cumsum(dim=1) % 2
    return bits, labels


def _cosine_lr(step: int, total: int, base: float, warmup_frac: float) -> float:
    """ParaRNN App. C: 10% warmup, cosine to 0."""
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return base * float(step + 1) / float(warm)
    t = float(step - warm) / float(max(total - warm, 1))
    return base * 0.5 * (1.0 + math.cos(math.pi * t))


class _S6Block(nn.Module):
    """Pre-norm residual diagonal selective SSM (S4D-Real).

    h_t = exp(-Δ_t A) ⊙ h_{t-1} + (B_t ⊙ x_t). Δ range Gu & Dao 2023
    [1e-3, 1e-1]. scan_diag eager so Autograd flows.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_model)
        self.C_proj = nn.Linear(d_model, d_model)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_model + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.linspace(math.log(1e-3), math.log(1e-1), d_model))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        dt = F.softplus(self.dt_proj(z) + self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.log_A))
        drive = self.B_proj(z) * z
        h = scan_diag(decay, drive, backend="eager")
        return x + self.C_proj(h)


class _ResidualSLSTM(nn.Module):
    def __init__(
        self,
        d_h: int,
        *,
        solver: str,
        newton_cfg: NewtonConfig,
        max_recurrent_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_h)
        self.rnn = ParaRNN(
            ParaSLSTM(d_h, d_h, mix="diag", max_recurrent_norm=max_recurrent_norm),
            config=newton_cfg,
            output_hidden=True,
            solver=solver,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.rnn(self.norm(x))


class _ParityNet(nn.Module):
    def __init__(
        self,
        d_h: int,
        num_layers: int,
        arm: str,
        newton_cfg: NewtonConfig,
        max_recurrent_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.embed = nn.Embedding(VOCAB, d_h)
        if arm == "ssm":
            self.blocks = nn.ModuleList(_S6Block(d_h) for _ in range(num_layers))
        else:
            solver = "auto" if arm == "newton" else "sequential"
            self.blocks = nn.ModuleList(
                _ResidualSLSTM(
                    d_h,
                    solver=solver,
                    newton_cfg=newton_cfg,
                    max_recurrent_norm=max_recurrent_norm,
                )
                for _ in range(num_layers)
            )
        self.head = nn.Linear(d_h, VOCAB)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def newton_residuals(self) -> list[float]:
        if self.arm != "newton":
            return []
        out: list[float] = []
        for block in self.blocks:
            if isinstance(block, _ResidualSLSTM):
                out.extend(st.max_residual for st in block.rnn.last_stats)
        return out


@torch.no_grad()
def _eval_acc(model: nn.Module, bits: Tensor, labels: Tensor) -> dict[str, float]:
    was_train = model.training
    model.eval()
    hit = model(bits).argmax(dim=-1) == labels
    out = {
        "tok": float(hit.float().mean()),
        "exact": float(hit.all(dim=1).float().mean()),
        "tok_t0": float(hit[:, 0].float().mean()),
        "tok_last": float(hit[:, -1].float().mean()),
    }
    model.train(was_train)
    return out


def _train_arm(device: torch.device, arm: str) -> dict:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    cfg = NewtonConfig(max_iters=NEWTON_ITERS, scan_backend=SCAN_BACKEND)
    model = _ParityNet(D_H, NUM_LAYERS, arm, cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    gen = torch.Generator().manual_seed(SEED)
    eval_gen = torch.Generator().manual_seed(SEED + 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arm={arm} n_params={n_params} lr={LR}")

    eval_bits, eval_lab = sample_parity(BATCH * 4, SEQ_LEN, generator=eval_gen)
    long_bits, long_lab = sample_parity(BATCH * 4, EVAL_SEQ_LEN, generator=eval_gen)
    eval_bits, eval_lab = eval_bits.to(device), eval_lab.to(device)
    long_bits, long_lab = long_bits.to(device), long_lab.to(device)

    loss0 = loss_f = 0.0
    curve_steps: list[int] = []
    eval_last_hist: list[float] = []
    long_last_hist: list[float] = []
    for step in range(STEPS):
        bits, labels = sample_parity(BATCH, SEQ_LEN, generator=gen)
        bits, labels = bits.to(device), labels.to(device)
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, STEPS, LR, WARMUP_FRAC)
        logits = model(bits)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_f = loss.item()
        if step == 0:
            loss0 = loss_f
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == STEPS:
            ev = _eval_acc(model, eval_bits, eval_lab)
            lg = _eval_acc(model, long_bits, long_lab)
            curve_steps.append(step)
            eval_last_hist.append(ev["tok_last"])
            long_last_hist.append(lg["tok_last"])
            res = model.newton_residuals()
            print(
                f"arm={arm} step={step:04d} ce={loss_f:.4f} "
                f"eval_last={ev['tok_last']:.3f} long_last={lg['tok_last']:.3f} "
                f"res={[f'{r:.2e}' for r in res] if res else '-'}"
            )
    ev = _eval_acc(model, eval_bits, eval_lab)
    lg = _eval_acc(model, long_bits, long_lab)
    return {
        "n_params": n_params,
        "loss0": loss0,
        "loss_final": loss_f,
        "eval_tok": ev["tok"],
        "eval_exact": ev["exact"],
        "eval_t0": ev["tok_t0"],
        "eval_last": ev["tok_last"],
        "long_tok": lg["tok"],
        "long_exact": lg["exact"],
        "long_last": lg["tok_last"],
        "curve_steps": curve_steps,
        "eval_last_hist": eval_last_hist,
        "long_last_hist": long_last_hist,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "parity smoke expects a CUDA GPU"
    print(f"parity start: {torch.cuda.get_device_name(device)} lr={LR}")

    curves: dict[str, dict] = {}
    for arm in ARMS:
        used = _train_arm(device, arm)
        if used["eval_tok"] <= 0.6:
            print(
                f"warning: arm={arm} eval_tok={used['eval_tok']:.3f} "
                f"last={used['eval_last']:.3f} still near chance"
            )
        curves[arm] = {
            "steps": used["curve_steps"],
            "eval_last_t16": used["eval_last_hist"],
            "eval_last_t32": used["long_last_hist"],
            "n_params": used["n_params"],
            "eval_last": used["eval_last"],
            "long_last": used["long_last"],
            "loss0": used["loss0"],
            "loss_final": used["loss_final"],
            "lr_used": LR,
        }
        print(
            f"arm={arm} done eval_tok={used['eval_tok']:.3f} "
            f"last={used['eval_last']:.3f} long_last={used['long_last']:.3f}"
        )

    curve_path = "parity_curves.json"
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump({"gpu": torch.cuda.get_device_name(device), "arms": curves}, f, indent=2)
        f.write("\n")
    print(f"wrote {curve_path}")

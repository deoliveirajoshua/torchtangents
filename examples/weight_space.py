"""
Train a small MLP on MNIST and compare it to its linearization (first-order
Taylor expansion around initialization).

- Full model: θ is trained with momentum on minibatches.
- Linearized model: we keep θ0 fixed and train Δθ in f_lin(x) = f(θ0, x) + J_θ0(x) · Δθ
  using autograd.functional.jvp for efficient Jacobian-vector products.

Requirements:
  pip install torch torchvision
"""

from __future__ import annotations
import math
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.stateless import functional_call
from torch.autograd.functional import jvp


# -----------------------
# Config (mirrors the JAX example)
# -----------------------
LEARNING_RATE = 1.0
MOMENTUM = 0.9
BATCH_SIZE = 128
TRAIN_EPOCHS = 10
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Model
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim=784, width=512, out_dim=10, w_std=1.0, b_std=0.05):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, out_dim)
        # Init like the JAX stax.Dense(·, W_std, b_std)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                nn.init.normal_(layer.weight, mean=0.0, std=w_std / math.sqrt(layer.weight.shape[1]))
                nn.init.normal_(layer.bias, mean=0.0, std=b_std)

    def forward(self, x):
        # x: [B, 1, 28, 28] or [B, 784]
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.erf(x)  # stax.Erf()
        x = self.fc2(x)
        return x


# -----------------------
# Helpers to “linearize” a model around θ0
# -----------------------
def named_params_tuple(model: nn.Module) -> Tuple[List[str], Tuple[torch.Tensor, ...]]:
    """Return names and a tuple of parameter tensors in a fixed (stable) order."""
    od = OrderedDict((n, p) for n, p in model.named_parameters())
    names = list(od.keys())
    params_tuple = tuple(od.values())
    return names, params_tuple


def tuple_to_param_dict(names: List[str], params_tuple: Tuple[torch.Tensor, ...]) -> OrderedDict:
    return OrderedDict((n, p) for n, p in zip(names, params_tuple))


class LinearizedModel(nn.Module):
    """
    f_lin(x; Δθ) = f(θ0, x) + J_{θ0}(x) · Δθ

    We implement Δθ as independent trainable tensors (same shapes as θ0).
    The forward pass uses:
      y0 = f(θ0, x)  (no grad through θ0)
      jvp_out = JVP_{θ0}[Δθ](x)
      return y0 + jvp_out
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model  # already initialized at θ0
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Snapshot θ0
        self.names, theta0_tuple = named_params_tuple(self.base)
        self.theta0_tuple = tuple(p.detach().clone() for p in theta0_tuple)

        # Create Δθ parameters (initialized to zeros)
        self.delta_params = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(p)) for p in self.theta0_tuple]
        )

    def forward(self, x):
        x = x if x.dim() == 2 else x.view(x.size(0), -1)

        # y0 at θ0 (no grad through θ0)
        with torch.no_grad():
            params0 = tuple_to_param_dict(self.names, self.theta0_tuple)
            y0 = functional_call(self.base, params0, (x,))

        # Define function of θ so we can take JVP at θ0
        def f_of_params(params_tuple: Tuple[torch.Tensor, ...]):
            p_dict = tuple_to_param_dict(self.names, params_tuple)
            return functional_call(self.base, p_dict, (x,))

        delta_tuple = tuple(self.delta_params)
        # jvp returns (f(θ0), Jv); we only need Jv since we already computed y0.
        _, jv = jvp(f_of_params, (self.theta0_tuple,), (delta_tuple,), create_graph=False, strict=True)
        return y0 + jv


# -----------------------
# Data
# -----------------------
def get_dataloaders(batch_size=BATCH_SIZE):
    tfm = transforms.Compose([
        transforms.ToTensor(),             # [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# -----------------------
# Train / Eval
# -----------------------
@torch.no_grad()
def eval_summary(model: nn.Module, loader: DataLoader, device=DEVICE):
    model.eval()
    total = 0
    ce_sum = 0.0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="sum")
        ce_sum += ce.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return dict(loss=ce_sum / total, acc=correct / total)


def main():
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_dataloaders()

    # Build base model at θ0 (same init used for both models)
    base = MLP().to(DEVICE)
    # Full model (train θ)
    full = MLP().to(DEVICE)
    full.load_state_dict(base.state_dict())  # start from same θ0

    # Linearized model (train Δθ only)
    lin = LinearizedModel(MLP().to(DEVICE))
    lin.base.load_state_dict(base.state_dict())  # use the same θ0 snapshot

    # Two optimizers: one for θ (full), one for Δθ (linearized)
    opt_full = torch.optim.SGD(full.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    opt_lin  = torch.optim.SGD(lin.delta_params.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    steps_per_epoch = 50000 // BATCH_SIZE  # to match original logging cadence
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    print("Training.")
    print("Epoch\tLoss(full)\tLoss(lin)")
    print("------------------------------------------")

    full.train(); lin.train()
    step = 0
    epoch = 0
    for ep in range(TRAIN_EPOCHS):
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # ---- full model step ----
            opt_full.zero_grad(set_to_none=True)
            logits_full = full(x)
            loss_full = F.cross_entropy(logits_full, y)
            loss_full.backward()
            opt_full.step()

            # ---- linearized model step ----
            opt_lin.zero_grad(set_to_none=True)
            logits_lin = lin(x)
            loss_lin = F.cross_entropy(logits_lin, y)
            loss_lin.backward()
            opt_lin.step()

            # Logging cadence similar to the JAX print frequency
            if step % steps_per_epoch == 0:
                print(f"{epoch}\t{loss_full.item():.4f}\t\t{loss_lin.item():.4f}")
                epoch += 1
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    # Final summaries (match the JAX script’s printout idea)
    # Use a larger slice for "train" eval (here: whole training loader with eval_summary)
    tr_full = eval_summary(full, train_loader)
    tr_lin  = eval_summary(lin,  train_loader)
    te_full = eval_summary(full, test_loader)
    te_lin  = eval_summary(lin,  test_loader)

    def pfx(tag, s_full, s_lin):
        print(f"\n[{tag}]")
        print(f"Full: loss={s_full['loss']:.4f}  acc={100*s_full['acc']:.2f}%")
        print(f"Lin.: loss={s_lin['loss']:.4f}  acc={100*s_lin['acc']:.2f}%")

    pfx("train", tr_full, tr_lin)
    pfx("test",  te_full, te_lin)


if __name__ == "__main__":
    main()

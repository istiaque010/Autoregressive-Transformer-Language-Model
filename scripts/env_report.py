"""Phase 1: hardware / backend report plus a CPU training smoke test (forward, backward, optimizer step).

Usage: python scripts/env_report.py  -> prints and writes results/metrics/env_report.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.model import GPT, GPTConfig, count_parameters  # noqa: E402
from src.utils import env_info, resolve_device  # noqa: E402


def main():
    device = resolve_device("auto")
    info = env_info(device)
    a, b = torch.randn(2, 3), torch.randn(3, 2)
    info["matmul_ok"] = tuple((a @ b).shape) == (2, 2)

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 512, (4, 33), device=device)
    losses = []
    for _ in range(20):  # overfit one batch: loss must go down
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    info["smoke_params"] = count_parameters(model)["total"]
    info["smoke_loss_first_last"] = [round(losses[0], 3), round(losses[-1], 3)]
    info["smoke_ok"] = losses[-1] < losses[0]
    info["gpu_backend_status"] = (
        f"GPU via torch.cuda ({info.get('gpu_name')})" if torch.cuda.is_available()
        else "CPU only (no CUDA/ROCm device visible to PyTorch)"
    )

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/env_report.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

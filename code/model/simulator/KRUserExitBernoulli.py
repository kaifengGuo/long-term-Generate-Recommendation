import torch
import torch.nn as nn


class _ExitBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class KRUserExitBernoulli(nn.Module):
    """
    Predict per-step leave probability p(exit|state, response) for Bernoulli sampling.
    """

    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        blocks = []
        prev = int(input_dim)
        for h in hidden_dims:
            h = int(h)
            blocks.append(_ExitBlock(prev, h, float(dropout)))
            prev = h
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(prev, 1)

    def forward(self, x):
        h = x
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def load_from_checkpoint(ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        input_dim = int(ckpt["input_dim"])
        hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", [128, 64]))
        dropout = float(ckpt.get("dropout", 0.1))
        model = KRUserExitBernoulli(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model = model.to(device)
        model.eval()
        spec = {
            "input_dim": input_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
            "response_cols": ckpt.get("response_cols", []),
            "user_feat_dim": int(ckpt.get("user_feat_dim", 0)),
            "max_hist_seq_len": int(ckpt.get("max_hist_seq_len", 50)),
            "recent_windows": ckpt.get("recent_windows", [5, 10]),
            "engage_weights": ckpt.get("engage_weights", [1.0, 0.7, 0.5, 0.5, 0.5, 0.5, -0.2]),
        }
        return model, spec

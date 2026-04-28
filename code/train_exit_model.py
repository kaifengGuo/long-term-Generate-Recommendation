import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

import utils
from model.simulator.KRUserExitBernoulli import KRUserExitBernoulli


RESP_COLS = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
USER_FEATS = [
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat6",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--user_meta_file", type=str, required=True)
    parser.add_argument("--data_separator", type=str, default=",")
    parser.add_argument("--meta_file_sep", type=str, default=",")
    parser.add_argument("--max_hist_seq_len", type=int, default=50)
    parser.add_argument("--recent_windows", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--engage_weights", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.5, 0.5, 0.5, -0.2])
    parser.add_argument("--val_user_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--early_step_max", type=int, default=5)
    parser.add_argument("--early_pos_weight", type=float, default=1.0)
    parser.add_argument("--early_neg_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()


def build_user_feature_map(user_meta_file, sep):
    user_df = pd.read_csv(user_meta_file, sep=sep).fillna("unknown")
    selected = [f for f in USER_FEATS if f in user_df.columns]
    if not selected:
        raise ValueError("No user features found for exit model.")

    user_vocab = utils.get_onehot_vocab(user_df, selected)
    feat_dim = sum(len(next(iter(user_vocab[f].values()))) for f in selected)

    user_map = {}
    for _, row in user_df.iterrows():
        uid = int(row["user_id"])
        vecs = []
        for f in selected:
            raw = row[f]
            vec = user_vocab[f].get(raw, None)
            if vec is None:
                vec = np.zeros(len(next(iter(user_vocab[f].values()))), dtype=np.float32)
            vecs.append(np.asarray(vec, dtype=np.float32))
        user_map[uid] = np.concatenate(vecs, axis=0).astype(np.float32)
    return user_map, feat_dim, selected


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def build_leave_dataset(args, user_map, user_feat_dim):
    use_cols = ["user_id", "session", "time_ms"] + RESP_COLS
    df = pd.read_csv(args.train_file, sep=args.data_separator, usecols=use_cols)
    df = df.sort_values(["user_id", "session", "time_ms"]).reset_index(drop=True)

    uids = df["user_id"].astype(np.int64).to_numpy()
    sess = df["session"].astype(np.int64).to_numpy()
    resp = df[RESP_COLS].astype(np.float32).to_numpy()
    n, rdim = resp.shape

    leave_label = np.ones(n, dtype=np.float32)
    if n > 1:
        same_next = (uids[:-1] == uids[1:]) & (sess[:-1] == sess[1:])
        leave_label[:-1] = (~same_next).astype(np.float32)

    hist_len_norm = np.zeros((n, 1), dtype=np.float32)
    hist_avg = np.zeros((n, rdim), dtype=np.float32)
    prev1 = np.zeros((n, rdim), dtype=np.float32)
    recent_windows = sorted(set(int(w) for w in args.recent_windows if int(w) > 0))
    recent = {w: np.zeros((n, rdim), dtype=np.float32) for w in recent_windows}

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = (uids[1:] != uids[:-1]) | (sess[1:] != sess[:-1])
    start_idx = np.where(starts)[0]
    end_idx = np.r_[start_idx[1:], n]
    lengths = end_idx - start_idx
    step_idx = np.arange(n, dtype=np.int64) - np.repeat(start_idx, lengths)

    for s, e in zip(start_idx, end_idx):
        rs = resp[s:e]
        L = rs.shape[0]
        idx = np.arange(L, dtype=np.float32)
        hist_len_norm[s:e, 0] = np.minimum(idx, float(args.max_hist_seq_len)) / float(max(1, args.max_hist_seq_len))

        if L <= 1:
            continue

        csum = np.cumsum(rs, axis=0, dtype=np.float32)
        hist_avg[s + 1 : e] = csum[:-1] / idx[1:, None]
        prev1[s + 1 : e] = rs[:-1]

        prefix = np.vstack([np.zeros((1, rdim), dtype=np.float32), csum])
        j = np.arange(L, dtype=np.int64)
        for w in recent_windows:
            a = np.maximum(0, j - w)
            b = j
            sums = prefix[b] - prefix[a]
            den = np.maximum(1, (b - a)).astype(np.float32).reshape(-1, 1)
            recent[w][s:e] = sums / den

    uniq_uids, inv = np.unique(uids, return_inverse=True)
    uniq_mat = np.zeros((len(uniq_uids), user_feat_dim), dtype=np.float32)
    zero = np.zeros((user_feat_dim,), dtype=np.float32)
    for i, uid in enumerate(uniq_uids):
        uniq_mat[i] = user_map.get(int(uid), zero)
    user_mat = uniq_mat[inv]

    weights = np.asarray(args.engage_weights, dtype=np.float32).reshape(1, -1)
    weights = weights[:, :rdim]
    hist_score = (hist_avg[:, : weights.shape[1]] * weights).sum(axis=1, keepdims=True)
    cur_score = (resp[:, : weights.shape[1]] * weights).sum(axis=1, keepdims=True)
    score_delta = cur_score - hist_score

    feat_blocks = [user_mat, hist_len_norm, hist_avg, prev1]
    for w in recent_windows:
        feat_blocks.append(recent[w])
    feat_blocks.extend([resp, hist_score, cur_score, score_delta])
    x = np.concatenate(feat_blocks, axis=1).astype(np.float32)
    y = leave_label.astype(np.float32)
    sample_weight = np.ones(n, dtype=np.float32)
    if args.early_step_max >= 0:
        early_mask = step_idx <= int(args.early_step_max)
        if args.early_pos_weight != 1.0:
            sample_weight[early_mask & (y > 0.5)] *= float(args.early_pos_weight)
        if args.early_neg_weight != 1.0:
            sample_weight[early_mask & (y <= 0.5)] *= float(args.early_neg_weight)

    spec = {
        "recent_windows": recent_windows,
        "engage_weights": args.engage_weights,
        "response_cols": RESP_COLS,
        "early_step_max": int(args.early_step_max),
        "early_pos_weight": float(args.early_pos_weight),
        "early_neg_weight": float(args.early_neg_weight),
    }
    return x, y, uids, spec, sample_weight


def eval_metrics(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    p = 1.0 / (1.0 + np.exp(-logits))
    bce = float(-(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))).mean())
    auc = _safe_auc(y, p)
    ap = _safe_ap(y, p)
    return {"bce": bce, "auc": auc, "ap": ap}


def main():
    args = parse_args()
    utils.set_random_seed(args.seed)

    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    print("[ExitModel] building user feature map")
    user_map, user_feat_dim, selected_user_feats = build_user_feature_map(args.user_meta_file, args.meta_file_sep)

    print("[ExitModel] building leave dataset")
    x, y, uids, spec, w = build_leave_dataset(args, user_map, user_feat_dim)
    print(f"[ExitModel] samples={len(y)}, feature_dim={x.shape[1]}, pos_rate={float(y.mean()):.4f}")

    uniq_users = np.unique(uids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq_users)
    n_val = max(1, int(len(uniq_users) * args.val_user_ratio))
    val_users = set(uniq_users[:n_val].tolist())
    val_mask = np.array([u in val_users for u in uids], dtype=bool)
    train_mask = ~val_mask

    x_train = torch.from_numpy(x[train_mask])
    y_train = torch.from_numpy(y[train_mask])
    w_train = torch.from_numpy(w[train_mask])
    x_val = torch.from_numpy(x[val_mask])
    y_val = torch.from_numpy(y[val_mask])

    train_loader = DataLoader(TensorDataset(x_train, y_train, w_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = KRUserExitBernoulli(input_dim=x.shape[1], hidden_dims=args.hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), device=device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    best_auc = -1.0
    best_bce = 1e9
    best_state = None
    train_losses = []
    val_reports = []

    gamma = float(args.focal_gamma)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = bce_loss(logits, yb)
            if gamma > 0:
                prob = torch.sigmoid(logits)
                pt = prob * yb + (1.0 - prob) * (1.0 - yb)
                mod = (1.0 - pt).pow(gamma)
                weighted = mod * loss_vec * wb
            else:
                weighted = loss_vec * wb
            loss = weighted.sum() / torch.clamp(wb.sum(), min=1e-6)
            loss.backward()
            optimizer.step()
            bs = xb.shape[0]
            loss_sum += float(loss.item()) * bs
            n += bs
        train_loss = loss_sum / max(n, 1)
        train_losses.append(train_loss)

        val_report = eval_metrics(model, val_loader, device)
        val_reports.append(val_report)
        print(
            f"[ExitModel] epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"val_bce={val_report['bce']:.6f} "
            f"val_auc={val_report['auc']:.6f} "
            f"val_ap={val_report['ap']:.6f}"
        )

        auc = val_report["auc"]
        bce = val_report["bce"]
        better = False
        if np.isfinite(auc) and auc > best_auc:
            better = True
        elif (not np.isfinite(auc)) and bce < best_bce:
            better = True
        if better:
            best_auc = auc
            best_bce = bce
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "state_dict": best_state,
        "input_dim": int(x.shape[1]),
        "hidden_dims": [int(v) for v in args.hidden_dims],
        "dropout": float(args.dropout),
        "response_cols": RESP_COLS,
        "user_feature_names": selected_user_feats,
        "user_feat_dim": int(user_feat_dim),
        "max_hist_seq_len": int(args.max_hist_seq_len),
        "recent_windows": spec["recent_windows"],
        "engage_weights": spec["engage_weights"],
        "early_step_max": spec["early_step_max"],
        "early_pos_weight": spec["early_pos_weight"],
        "early_neg_weight": spec["early_neg_weight"],
        "seed": int(args.seed),
    }
    torch.save(ckpt, args.model_path)
    print(f"[ExitModel] saved checkpoint to {args.model_path}")

    report = {
        "train_file": args.train_file,
        "user_meta_file": args.user_meta_file,
        "n_samples": int(len(y)),
        "feature_dim": int(x.shape[1]),
        "recent_windows": spec["recent_windows"],
        "engage_weights": spec["engage_weights"],
        "early_step_max": spec["early_step_max"],
        "early_pos_weight": spec["early_pos_weight"],
        "early_neg_weight": spec["early_neg_weight"],
        "train_pos_rate": float(y_train.float().mean().item()),
        "val_pos_rate": float(y_val.float().mean().item()),
        "best_val_auc": None if not np.isfinite(best_auc) else float(best_auc),
        "best_val_bce": float(best_bce),
        "train_losses": [float(v) for v in train_losses],
        "val_reports": val_reports,
    }
    report_path = args.model_path + ".report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[ExitModel] saved report to {report_path}")


if __name__ == "__main__":
    main()

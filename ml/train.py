"""
Train Bi-Level GAT on the synthetic dataset.

Reproduces the synthetic-dataset experiment from the paper:
  - 200 users, 100 activities, 5,000 interactions, 600 social edges
  - 80/20 temporal split
  - Targets: RMSE, NDCG@5/10/20, HR@10
  - Compares to MF and GCN baselines

Outputs:
  ml/checkpoint.pt — model weights
  ml/metrics.json — final metrics (used by README + frontend)
  data/embeddings.npz — user/item embeddings for the API to use
"""
import os
import json
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import BiLevelGAT  # noqa

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT_ML = ROOT / 'ml'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_data():
    users = []
    with open(DATA / 'users.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            users.append(r)
    items = []
    with open(DATA / 'activities.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            items.append(r)
    inters = []
    with open(DATA / 'interactions.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            inters.append(r)
    edges = []
    with open(DATA / 'social_edges.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            edges.append((int(r['user_a']), int(r['user_b'])))
    return users, items, inters, edges


def build_features(users, items):
    # User features: [age_norm, gender, tech, qol4]
    Xu = []
    Qu = []
    for u in users:
        age = (int(u['age']) - 60) / 25.0
        gender = 1.0 if u['gender'] == 'F' else 0.0
        tech = float(u['tech_literacy'])
        q = [float(u['baseline_qol_physical']) / 100,
             float(u['baseline_qol_psychological']) / 100,
             float(u['baseline_qol_social']) / 100,
             float(u['baseline_qol_environment']) / 100]
        Xu.append([age, gender, tech] + q)
        Qu.append(q)
    Xu = np.array(Xu, dtype=np.float32)
    Qu = np.array(Qu, dtype=np.float32)

    # Item features: one-hot category + cognitive_benefit + difficulty + duration
    cats = sorted({i['category'] for i in items})
    cat2idx = {c: k for k, c in enumerate(cats)}
    diff_map = {'easy': 0.33, 'medium': 0.66, 'hard': 1.0}
    Xi = []
    for it in items:
        v = [0.0] * len(cats)
        v[cat2idx[it['category']]] = 1.0
        v.append(float(it['cognitive_benefit']))
        v.append(diff_map[it['difficulty']])
        v.append(float(it['duration_min']) / 60.0)
        Xi.append(v)
    Xi = np.array(Xi, dtype=np.float32)
    return Xu, Qu, Xi


def split_temporal(inters, ratio=0.8):
    inters = sorted(inters, key=lambda r: r['timestamp'])
    n = len(inters)
    cut = int(n * ratio)
    return inters[:cut], inters[cut:]


def build_adj(N_u, edges):
    A = np.zeros((N_u, N_u), dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    A += np.eye(N_u, dtype=np.float32)  # self-loop
    return A


def build_ui_adj(N_u, N_i, train_inters):
    A = np.zeros((N_u, N_i), dtype=np.float32)
    for r in train_inters:
        A[int(r['user_id']), int(r['activity_id'])] = 1
    return A


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def ndcg_at_k(ranked_true_rel, k):
    """ranked_true_rel: list of relevance scores in predicted order."""
    rel = np.array(ranked_true_rel[:k], dtype=np.float32)
    if len(rel) == 0:
        return 0.0
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_at_k(ranked_relevant_flags, k):
    return float(any(ranked_relevant_flags[:k]))


def evaluate_ranking(model_predict_fn, test_inters, N_i, k_values=(5, 10, 20)):
    """Group test by user, rank all items, compute NDCG/HR."""
    by_user = {}
    for r in test_inters:
        u = int(r['user_id'])
        by_user.setdefault(u, []).append((int(r['activity_id']), float(r['rating'])))
    ndcg = {k: [] for k in k_values}
    hr10 = []
    for u, items_ratings in by_user.items():
        if not items_ratings:
            continue
        truth = {iid: rating for iid, rating in items_ratings}
        scores = model_predict_fn(u, list(range(N_i)))  # [N_i]
        order = np.argsort(-scores)
        ranked_relevance = [truth.get(int(i), 0.0) for i in order]
        ranked_relevant_flags = [int(i) in truth for i in order]
        for k in k_values:
            ndcg[k].append(ndcg_at_k(ranked_relevance, k))
        hr10.append(hit_at_k(ranked_relevant_flags, 10))
    return {f'NDCG@{k}': float(np.mean(v)) for k, v in ndcg.items()} | {
        'HR@10': float(np.mean(hr10))
    }


# ==================== Main ====================
print('=== Loading data ===')
users, items, inters, edges = load_data()
N_u = len(users)
N_i = len(items)
print(f'  users={N_u}, items={N_i}, interactions={len(inters)}, edges={len(edges)}')

Xu_np, Qu_np, Xi_np = build_features(users, items)
A_np = build_adj(N_u, edges)
train_inters, test_inters = split_temporal(inters, 0.8)
UIA_np = build_ui_adj(N_u, N_i, train_inters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  device={device}')

Xu = torch.tensor(Xu_np, device=device)
Xi = torch.tensor(Xi_np, device=device)
Qu = torch.tensor(Qu_np, device=device)
A = torch.tensor(A_np, device=device)
UIA = torch.tensor(UIA_np, device=device)

train_u = torch.tensor([int(r['user_id']) for r in train_inters],
                       dtype=torch.long, device=device)
train_i = torch.tensor([int(r['activity_id']) for r in train_inters],
                       dtype=torch.long, device=device)
train_r = torch.tensor([float(r['rating']) for r in train_inters],
                       dtype=torch.float32, device=device)
test_u = torch.tensor([int(r['user_id']) for r in test_inters],
                      dtype=torch.long, device=device)
test_i = torch.tensor([int(r['activity_id']) for r in test_inters],
                      dtype=torch.long, device=device)
test_r = torch.tensor([float(r['rating']) for r in test_inters],
                      dtype=torch.float32, device=device)


# ==================== Baseline 1: MF ====================
class MF(nn.Module):
    def __init__(self, N_u, N_i, d=32):
        super().__init__()
        self.U = nn.Embedding(N_u, d)
        self.V = nn.Embedding(N_i, d)
        self.bu = nn.Embedding(N_u, 1)
        self.bi = nn.Embedding(N_i, 1)
        self.global_b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_normal_(self.U.weight)
        nn.init.xavier_normal_(self.V.weight)

    def forward(self, u, i):
        x = (self.U(u) * self.V(i)).sum(-1)
        x = x + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1) + self.global_b
        return 1.0 + 4.0 * torch.sigmoid(x)


# ==================== Baseline 2: GCN ====================
class GCN(nn.Module):
    def __init__(self, du, di, d=64):
        super().__init__()
        self.up = nn.Linear(du, d)
        self.ip = nn.Linear(di, d)
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d)
        self.pred = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, 1)
        )

    def forward(self, Xu, Xi, A_norm, UIA, u_idx, i_idx):
        hu = F.relu(self.up(Xu))
        hi = F.relu(self.ip(Xi))
        # GCN update on social
        deg = A_norm.sum(-1, keepdim=True).clamp(min=1)
        hu = F.relu(self.W1(A_norm @ hu / deg))
        hu = self.W2(hu)
        # plus content message via UIA averaging
        deg_i = UIA.sum(-1, keepdim=True).clamp(min=1)
        msg = UIA @ hi / deg_i
        hu = hu + msg
        hu_p = hu[u_idx]
        hi_p = hi[i_idx]
        x = self.pred(torch.cat([hu_p, hi_p], -1)).squeeze(-1)
        return 1.0 + 4.0 * torch.sigmoid(x)


def train_loop(model, params_dict_fn, name, epochs=80, lr=1e-2, weight_decay=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best = float('inf')
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = params_dict_fn(model, train_u, train_i)
        loss = loss_fn(pred, train_r)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                p_test = params_dict_fn(model, test_u, test_i)
                rm = rmse(p_test.cpu().numpy(), test_r.cpu().numpy())
                if rm < best:
                    best = rm
                print(f'  [{name}] ep {ep+1}: train_loss={loss.item():.4f} '
                      f'test_rmse={rm:.4f}')
    return best


# Train MF
print('\n=== Train MF ===')
mf = MF(N_u, N_i, d=32).to(device)
mf_rmse = train_loop(mf, lambda m, u, i: m(u, i), 'MF', epochs=80)

# Train GCN
print('\n=== Train GCN ===')
gcn = GCN(Xu.shape[1], Xi.shape[1], d=64).to(device)
gcn_rmse = train_loop(
    gcn, lambda m, u, i: m(Xu, Xi, A, UIA, u, i), 'GCN', epochs=80
)

# Train Bi-Level GAT
print('\n=== Train Bi-Level GAT ===')
gat = BiLevelGAT(Xu.shape[1], Xi.shape[1], d_hidden=64, d_qol=4).to(device)


def gat_predict(u_idx, i_idx, model=gat):
    pred, _, _ = model(Xu, Xi, A, UIA, u_idx, i_idx, Qu)
    return pred


gat_rmse = train_loop(
    gat, lambda m, u, i: gat_predict(u, i, m), 'Bi-Level GAT', epochs=120, lr=5e-3
)

# Final eval & save
print('\n=== Final Evaluation ===')
gat.eval()
with torch.no_grad():
    pred_gat = gat_predict(test_u, test_i, gat).cpu().numpy()
    pred_mf = mf(test_u, test_i).cpu().numpy()
    pred_gcn = gcn(Xu, Xi, A, UIA, test_u, test_i).cpu().numpy()
    truth = test_r.cpu().numpy()

results = {
    'dataset': 'synthetic',
    'n_users': N_u,
    'n_items': N_i,
    'n_train': len(train_inters),
    'n_test': len(test_inters),
    'MF': {'RMSE': rmse(pred_mf, truth), 'params': sum(p.numel() for p in mf.parameters())},
    'GCN': {'RMSE': rmse(pred_gcn, truth), 'params': sum(p.numel() for p in gcn.parameters())},
    'BiLevelGAT': {'RMSE': rmse(pred_gat, truth), 'params': sum(p.numel() for p in gat.parameters())},
}


def predict_for_user(u, all_items, model):
    u_t = torch.tensor([u] * len(all_items), dtype=torch.long, device=device)
    i_t = torch.tensor(all_items, dtype=torch.long, device=device)
    with torch.no_grad():
        return gat_predict(u_t, i_t, model).cpu().numpy()


print('  Computing ranking metrics for Bi-Level GAT...')
rank = evaluate_ranking(
    lambda u, items: predict_for_user(u, items, gat),
    test_inters, N_i,
)
results['BiLevelGAT'].update(rank)

# improvement %
mf_r = results['MF']['RMSE']
gcn_r = results['GCN']['RMSE']
gat_r = results['BiLevelGAT']['RMSE']
results['improvement'] = {
    'rmse_reduction_vs_MF_pct': round((mf_r - gat_r) / mf_r * 100, 2),
    'rmse_reduction_vs_GCN_pct': round((gcn_r - gat_r) / gcn_r * 100, 2),
}

print('\n=== Results ===')
for k, v in results.items():
    print(f'  {k}: {v}')

# Save model checkpoint and metrics
OUT_ML.mkdir(exist_ok=True)
torch.save({
    'state_dict': gat.state_dict(),
    'config': {
        'd_u_in': Xu.shape[1],
        'd_i_in': Xi.shape[1],
        'd_hidden': 64,
        'd_qol': 4,
    },
}, OUT_ML / 'checkpoint.pt')

with open(OUT_ML / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save embeddings for the API to load (avoid retraining)
gat.eval()
with torch.no_grad():
    h_u_out, h_i_out, _, _ = gat.encode(Xu, Xi, A, UIA, Qu)
np.savez(
    DATA / 'embeddings.npz',
    user_emb=h_u_out.cpu().numpy(),
    item_emb=h_i_out.cpu().numpy(),
)

print(f'\n✅ Training complete. Saved checkpoint, metrics, embeddings.')

import os, random, copy, torch, timm, cv2
import numpy as np, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt, seaborn as sns

# ─────────────────── CONFIG & SETUP ───────────────────
SPLITS_DIR, SAVE_DIR, RESULTS_DIR = r"D:\RL\splits", r"D:\RL\models_new", r"D:\RL\results_new"
os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
CLASSES = ["Stage_0 (Unripe)", "Stage_1(Early Ripe)", "Stage_2 (Partially Ripe)", "Stage_3 (Ripe)"]
NUM_CLASSES, IMG_SIZE, PHI, SEED = 4, 224, 1.618, 42
WOLVES, ITERATIONS, PROXY_EPOCHS, FULL_EPOCHS = 5, 8, 4, 35
RL_A_VALUES = [0.5, PHI, 3.0]  # Exploit (0.5), Golden Balance (1.618), Explore (3.0)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")

tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class MangoDataset(Dataset):
    def __init__(self, root):
        self.samples, self.labels = [], []
        for lbl, cls in enumerate(CLASSES):
            p = os.path.join(root, cls)
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(('.jpg', '.png')): self.samples.append(os.path.join(p, f)); self.labels.append(lbl)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return tf(Image.open(self.samples[i]).convert("RGB")), self.labels[i]

def build_swin_t(dropout=0.2):
    bb = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0, global_pool="")
    class SwinWithHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb, self.pool = bb, nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES))
        def forward(self, x):
            f = self.bb(x)
            if f.dim() == 4: f = f.view(f.shape[0], -1, 768)
            return self.head(self.pool(f.permute(0, 2, 1)).squeeze(-1))
    m = SwinWithHead().to(DEVICE)
    for p in m.parameters(): p.requires_grad = False
    for p in m.head.parameters(): p.requires_grad = True
    for n, p in m.bb.named_parameters():
        if "layers.2" in n or "layers.3" in n: p.requires_grad = True
    return m

def train_epoch(model, loader, opt, crit):
    model.train(); total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step()
        total_loss += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); total += len(y)
    return total_loss / total, correct / total

def eval_model(model, loader, crit):
    model.eval(); total_loss, preds, targets, probs = 0.0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); pr = torch.softmax(out, dim=1)
            total_loss += crit(out, y).item() * len(y)
            preds.extend(out.argmax(1).cpu().numpy()); targets.extend(y.cpu().numpy()); probs.extend(pr.cpu().numpy())
    acc = np.mean(np.array(preds) == np.array(targets))
    return total_loss / len(targets), acc, f1_score(targets, preds, average="macro"), targets, preds, np.array(probs)

class RLMetaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x): return torch.softmax(self.net(x), dim=-1)

def get_stratified_subset(ds, n_per_class=100):
    idx, lbs = [], np.array(ds.labels)
    for c in range(NUM_CLASSES):
        c_idx = np.where(lbs == c)[0]
        idx.extend(np.random.choice(c_idx, min(n_per_class, len(c_idx)), replace=False).tolist())
    return torch.utils.data.Subset(ds, idx)

def proxy_train(hp, tr_sub, va_sub):
    m = build_swin_t(hp["dropout"])
    hp_p = [p for n, p in m.named_parameters() if p.requires_grad and "head" in n]
    bb_p = [p for n, p in m.named_parameters() if p.requires_grad and "head" not in n]
    opt = torch.optim.AdamW([{"params": hp_p, "lr": hp["lr"]}, {"params": bb_p, "lr": hp["lr"] * 0.1}], weight_decay=hp["wd"])
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_ld, va_ld = DataLoader(tr_sub, batch_size=int(hp["bs"]), shuffle=True), DataLoader(va_sub, batch_size=int(hp["bs"]), shuffle=False)
    best = 0.0
    for _ in range(PROXY_EPOCHS):
        train_epoch(m, tr_ld, opt, crit)
        _, acc, _, _, _, _ = eval_model(m, va_ld, crit); best = max(best, acc)
    del m; torch.cuda.empty_cache()
    return best

def rl_gwo_search(tr_ds, va_ds, meta_episodes=3):
    print("\n" + "=" * 55 + "\nRL-GUIDED φ-MGWO HYPERPARAMETER SEARCH\n" + "=" * 55)
    tr_sub, va_sub = get_stratified_subset(tr_ds, 100), get_stratified_subset(va_ds, 20)
    meta_net, meta_opt = RLMetaController(), torch.optim.Adam(RLMetaController().parameters(), lr=1e-2)
    best_hp, best_acc = None, -1.0
    rnd = lambda: {"lr": 10**np.random.uniform(-5,-3), "dropout": np.random.uniform(0.1,0.5), "wd": 10**np.random.uniform(-5,-3), "bs": float(np.random.choice([16,32]))}

    for ep in range(1, meta_episodes + 1):
        wolves = [rnd() for _ in range(WOLVES)]
        fit = [proxy_train(w, tr_sub, va_sub) for w in wolves]
        hist = {"a": [], "b": [], "d": []}; traj = []

        for it in range(ITERATIONS):
            st = torch.tensor([it/ITERATIONS, max(fit), np.mean(fit), np.std(fit), max(fit)-min(fit), float(np.std(fit) < 0.005)], dtype=torch.float32)
            dist = torch.distributions.Categorical(meta_net(st))
            act = dist.sample(); traj.append((st, dist.log_prob(act))); a = RL_A_VALUES[act.item()]

            s_idx = np.argsort(fit)[::-1]
            a_w, b_w, d_w = wolves[s_idx[0]], wolves[s_idx[1]], wolves[s_idx[2]]
            for k, w in zip(["a", "b", "d"], [fit[s_idx[0]], fit[s_idx[1]], fit[s_idx[2]]]):
                hist[k].append(w); hist[k] = hist[k][-4:]
            
            inv_v = {k: 1.0 / (np.var(hist[k]) + 1e-12 if len(hist[k]) > 1 else 1.0) for k in hist}
            tot = sum(inv_v.values()); wts = {k: inv_v[k] / tot for k in inv_v}

            for i in range(WOLVES):
                if i in s_idx[:3]: continue
                new_w = {}
                for k in ["lr", "dropout", "wd"]:
                    step = lambda ldr: ldr[k] - (2*a*np.random.rand()-a) * abs(2*np.random.rand()*ldr[k] - wolves[i][k])
                    new_w[k] = wts["a"]*step(a_w) + wts["b"]*step(b_w) + wts["d"]*step(d_w)
                new_w["lr"] = float(np.clip(new_w["lr"], 1e-5, 1e-3))
                new_w["dropout"] = float(np.clip(new_w["dropout"], 0.1, 0.5))
                new_w["wd"] = float(np.clip(new_w["wd"], 1e-5, 1e-3))
                new_w["bs"] = float(np.random.choice([16, 32]))
                n_fit = proxy_train(new_w, tr_sub, va_sub)
                if n_fit > fit[i]: wolves[i], fit[i] = new_w, n_fit

            print(f"  Ep {ep}/{meta_episodes} | Iter {it+1}/{ITERATIONS} | Action: a={a:.2f} | Best Acc: {max(fit):.4f}")

        loss = -sum(lp * (max(fit) - 0.3) for _, lp in traj) / len(traj)
        meta_opt.zero_grad(); loss.backward(); meta_opt.step()
        if max(fit) > best_acc: best_acc, best_hp = max(fit), copy.deepcopy(wolves[int(np.argmax(fit))])

    torch.save(meta_net.state_dict(), os.path.join(SAVE_DIR, "rl_meta_controller.pth"))
    print(f"\n✅ Winning HP: lr={best_hp['lr']:.6f}, dropout={best_hp['dropout']:.2f}, wd={best_hp['wd']:.6f}, bs={int(best_hp['bs'])}")
    return best_hp

def full_train(hp, tr_ds, va_ds, te_ds):
    print("\n" + "=" * 55 + "\nFULL SWIN-T TRAINING\n" + "=" * 55)
    tr_ld, va_ld, te_ld = DataLoader(tr_ds, batch_size=int(hp["bs"]), shuffle=True), DataLoader(va_ds, batch_size=int(hp["bs"]), shuffle=False), DataLoader(te_ds, batch_size=int(hp["bs"]), shuffle=False)
    m = build_swin_t(hp["dropout"])
    head_p = [p for n, p in m.named_parameters() if p.requires_grad and "head" in n]
    bb_p   = [p for n, p in m.named_parameters() if p.requires_grad and "head" not in n]
    opt = torch.optim.AdamW([{"params": head_p, "lr": hp["lr"]}, {"params": bb_p, "lr": hp["lr"] * 0.1}], weight_decay=hp["wd"])
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EPOCHS)

    best_acc, best_sd, pat = 0.0, None, 0
    for ep in range(1, FULL_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(m, tr_ld, opt, crit)
        va_loss, va_acc, va_f1, _, _, _ = eval_model(m, va_ld, crit)
        sched.step()
        print(f"Epoch [{ep:02d}/{FULL_EPOCHS}] Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f} | Val F1: {va_f1:.4f}")
        if va_acc > best_acc: best_acc, best_sd, pat = va_acc, copy.deepcopy(m.state_dict()), 0
        else:
            pat += 1
            if pat >= 8: print(f"⏹ Early stopping at epoch {ep}"); break

    torch.save(best_sd, os.path.join(SAVE_DIR, "swin_t_rl_mgwo_best.pth"))
    m.load_state_dict(best_sd)
    te_loss, te_acc, te_f1, targets, preds, probs = eval_model(m, te_ld, crit)

    print("\n" + "=" * 55 + f"\nTEST RESULTS: Accuracy={te_acc*100:.2f}%, Macro F1={te_f1:.4f}\n" + "=" * 55)
    print(classification_report(targets, preds, target_names=CLASSES, digits=4))

    # Confusion matrix & ROC Curves
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(targets, preds), annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix (Test Set)"); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_test.png"), dpi=150); plt.close()

    auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC: {auc:.4f}")
    plt.figure(figsize=(7, 5))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((np.array(targets) == i).astype(int), probs[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC={roc_auc_score((np.array(targets) == i).astype(int), probs[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.title(f"ROC Curves (Macro AUC = {auc:.4f})")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves_test.png"), dpi=150); plt.close()

if __name__ == "__main__":
    tr_ds, va_ds, te_ds = MangoDataset(os.path.join(SPLITS_DIR, "train")), MangoDataset(os.path.join(SPLITS_DIR, "val")), MangoDataset(os.path.join(SPLITS_DIR, "test"))
    print(f"Loaded: Train={len(tr_ds)}, Val={len(va_ds)}, Test={len(te_ds)}")
    best_hp = rl_gwo_search(tr_ds, va_ds, meta_episodes=3)
    full_train(best_hp, tr_ds, va_ds, te_ds)

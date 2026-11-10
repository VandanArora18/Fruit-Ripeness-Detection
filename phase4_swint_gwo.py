import os
import random
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────
SPLITS_DIR  = r"D:\RL\splits"
SAVE_DIR    = r"D:\RL\models_new"
RESULTS_DIR = r"D:\RL\results_new"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES     = [
    "Stage_0 (Unripe)",
    "Stage_1(Early Ripe)",
    "Stage_2 (Partially Ripe)",
    "Stage_3 (Ripe)",
]
NUM_CLASSES = 4
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────
# HYPERPARAMETERS (RL-GUIDED φ-MGWO + TRAINING)
# ─────────────────────────────────────────
WOLVES         = 5
ITERATIONS     = 8
PROXY_EPOCHS   = 8
FULL_EPOCHS    = 35
EARLY_PATIENCE = 8

PHI = 1.618   # φ-MGWO constant

# Rolling window length for leader accuracies (uncertainty weighting)
ROLL_WINDOW = 4

# RL meta-controller settings
RL_STATE_DIM  = 6    # [iter/max, best_acc, mean_acc, std_acc, spread, plateau]
RL_ACTION_DIM = 3    # 0=exploit(a=0.5), 1=balanced(a=PHI), 2=explore(a=3.0)
RL_A_VALUES   = [0.5, PHI, 3.0]   # 'a' values for GWO step, indexed by action

# ─────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────
class MangoDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.transform = transform
        self.samples   = []
        self.labels    = []

        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ─────────────────────────────────────────
# SWIN-T MODEL (UNFREEZE LAST STAGE)
# ─────────────────────────────────────────
def build_swin_t(num_classes, dropout=0.2):
    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=0,
        global_pool=""
    )

    class SwinWithHead(nn.Module):
        def __init__(self, backbone, in_features, num_classes, dropout):
            super().__init__()
            self.backbone = backbone
            self.pool     = nn.AdaptiveAvgPool1d(1)
            self.head     = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.backbone(x)        # (B, H, W, C) or (B, L, C)
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.view(B, H * W, C)
            if x.dim() == 3:
                x = x.permute(0, 2, 1)  # (B, C, L)
                x = self.pool(x).squeeze(-1)  # (B, C)
            return self.head(x)

    in_features = 768
    model       = SwinWithHead(backbone, in_features, num_classes, dropout)

    # 1) Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # 2) Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    # 3) Unfreeze LAST Swin stage (layers.3.*)
    for name, param in model.backbone.named_parameters():
        if name.startswith("layers.3"):
            param.requires_grad = True

    # Sanity check
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE)
        out   = model(dummy)
        assert out.shape == (2, num_classes), f"Bad output shape: {out.shape}"
    model.train()

    return model.to(DEVICE)

# ─────────────────────────────────────────
# TRAIN / EVAL UTILITIES
# ─────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────
# PROXY TRAINING FOR GWO EVALUATION
# ─────────────────────────────────────────
def get_stratified_subset(dataset, n_per_class, seed=42):
    """Pick an equal number of samples from each class for fair proxy evaluation."""
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.labels)
    chosen_indices = []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        n_select = min(n_per_class, len(idx))
        chosen = rng.choice(idx, size=n_select, replace=False)
        chosen_indices.extend(chosen.tolist())
    return torch.utils.data.Subset(dataset, chosen_indices)


def proxy_train(hp, train_subset, val_subset):
    lr           = float(hp["lr"])
    dropout      = float(hp["dropout"])
    weight_decay = float(hp["weight_decay"])
    batch_size   = int(hp["batch_size"])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_swin_t(NUM_CLASSES, dropout=dropout)

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params,     "lr": lr},
            {"params": backbone_params, "lr": lr * 0.1},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    for _ in range(PROXY_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion)
        _, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    del model
    torch.cuda.empty_cache()
    return best_val_acc


# ═══════════════════════════════════════════════════════════════════
#  RL META-CONTROLLER FOR φ-MGWO
#
#  The RL agent controls the exploration parameter 'a' of the GWO at
#  each iteration.  Instead of a fixed a=PHI (original) or the linear
#  decay a = 2 - 2t/T (classic GWO), the agent LEARNS when to
#  exploit (small a → wolves converge) vs explore (large a → spread).
#
#  State  : 6-dim vector describing the current GWO search landscape
#  Action : {0=exploit, 1=balanced(φ), 2=explore}
#  Reward : improvement in best wolf accuracy after the GWO iteration
#
#  Policy : tiny MLP trained with REINFORCE (online, one episode = full GWO run)
# ═══════════════════════════════════════════════════════════════════

class RLMetaController(nn.Module):
    """
    Policy network for controlling GWO's exploration parameter 'a'.
    Input  : 6-dim state
    Output : softmax probabilities over 3 actions
    """
    def __init__(self, state_dim=RL_STATE_DIM, hidden=32, num_actions=RL_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def build_gwo_state(iteration, max_iterations, fitness):
    """
    Build 6-dim state vector for RL meta-controller.
      0: normalised iteration (progress)
      1: best fitness (alpha accuracy)
      2: mean fitness
      3: std of fitness (diversity)
      4: spread = best - worst
      5: plateau flag (std < 0.005)
    """
    fitness_arr = np.array(fitness, dtype=np.float32)
    state = np.array([
        iteration / max(max_iterations, 1),   # progress [0,1]
        float(np.max(fitness_arr)),             # best acc
        float(np.mean(fitness_arr)),            # mean acc
        float(np.std(fitness_arr)),             # diversity
        float(np.max(fitness_arr) - np.min(fitness_arr)),  # spread
        1.0 if np.std(fitness_arr) < 0.005 else 0.0,       # plateau flag
    ], dtype=np.float32)
    return state


# ─────────────────────────────────────────
# RL-GUIDED φ-MGWO HYPERPARAMETER SEARCH
# ─────────────────────────────────────────
def rl_gwo_search(train_dataset, val_dataset, rl_meta_episodes=3):
    """
    Run RL-guided φ-MGWO for hyperparameter optimisation.

    Outer loop: rl_meta_episodes — each episode runs a full GWO and
                collects (state, action, reward) tuples, then updates
                the RL policy with REINFORCE.
    Inner loop: standard φ-MGWO wolf update with 'a' chosen by RL.

    Returns best_hp dict.
    """
    print("\n" + "=" * 60)
    print("RL-GUIDED φ-MGWO Hyperparameter Search")
    print(f"  Wolves           : {WOLVES}")
    print(f"  GWO Iterations   : {ITERATIONS}")
    print(f"  Proxy epochs     : {PROXY_EPOCHS}")
    print(f"  RL meta-episodes : {rl_meta_episodes}")
    print(f"  phi (ϕ)          : {PHI}")
    print(f"  RL action space  : a ∈ {RL_A_VALUES}")
    print("=" * 60)

    # ── Initialise RL meta-controller ──────────────────────────────
    meta_ctrl = RLMetaController().to("cpu")   # lightweight, runs on CPU
    meta_opt  = torch.optim.Adam(meta_ctrl.parameters(), lr=1e-2)
    rl_baseline = 0.0

    best_hp_overall   = None
    best_acc_overall  = -1.0

    def random_wolf():
        return {
            "lr":           10 ** np.random.uniform(-5, -3),
            "dropout":      np.random.uniform(0.1, 0.5),
            "weight_decay": 10 ** np.random.uniform(-5, -3),
            "batch_size":   float(np.random.choice([16, 32]))
        }

    # ── RL outer loop — run multiple GWO episodes ──────────────────
    for rl_ep in range(1, rl_meta_episodes + 1):
        print(f"\n{'─'*60}")
        print(f"RL Episode {rl_ep}/{rl_meta_episodes}")
        print(f"{'─'*60}")

        wolves  = [random_wolf() for _ in range(WOLVES)]
        fitness = [0.0] * WOLVES

        leader_hist = {"alpha": [], "beta": [], "delta": []}

        # Collect trajectory for REINFORCE
        trajectory = []   # list of (state_tensor, action, log_prob)

        # ── Evaluate initial wolves ─────────────────────────────────
        print("  Evaluating initial wolves...")
        for i, w in enumerate(wolves):
            fitness[i] = proxy_train(w, train_dataset, val_dataset)
            print(f"  Wolf {i+1} | lr={w['lr']:.2e} | dropout={w['dropout']:.2f} | "
                  f"wd={w['weight_decay']:.2e} | bs={int(w['batch_size'])} | "
                  f"val_acc={fitness[i]:.4f}")

        # ── GWO iterations with RL-controlled 'a' ──────────────────
        for it in range(ITERATIONS):
            # ── RL: build state, sample action ──────────────────────
            state_np = build_gwo_state(it, ITERATIONS, fitness)
            state_t  = torch.tensor(state_np, dtype=torch.float32)
            probs    = meta_ctrl(state_t)
            dist     = torch.distributions.Categorical(probs)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

            a = RL_A_VALUES[action.item()]   # chosen exploration parameter

            trajectory.append((state_t, action.item(), log_prob))

            print(f"\n  Iteration {it+1}/{ITERATIONS} | RL action: {action.item()} "
                  f"({'exploit' if action==0 else 'balanced-φ' if action==1 else 'explore'}) "
                  f"| a = {a:.3f}")

            # ── Sort wolves, identify leaders ───────────────────────
            sorted_idx = np.argsort(fitness)[::-1]
            alpha_idx, beta_idx, delta_idx = sorted_idx[0], sorted_idx[1], sorted_idx[2]

            alpha = wolves[alpha_idx]
            beta  = wolves[beta_idx]
            delta = wolves[delta_idx]

            # ── Update rolling history of leaders ───────────────────
            leader_hist["alpha"].append(fitness[alpha_idx])
            leader_hist["beta"].append(fitness[beta_idx])
            leader_hist["delta"].append(fitness[delta_idx])

            for k in leader_hist:
                if len(leader_hist[k]) > ROLL_WINDOW:
                    leader_hist[k] = leader_hist[k][-ROLL_WINDOW:]

            # ── Uncertainty-weighted leader influence ────────────────
            variances = {k: np.var(v) if len(v) > 1 else 0.0 for k, v in leader_hist.items()}
            eps       = 1e-12
            inv_var   = {k: 1.0 / (variances[k] + eps) for k in variances}
            total_inv = sum(inv_var.values())
            weights   = {k: inv_var[k] / total_inv for k in inv_var}

            # ── Update non-leader wolves ─────────────────────────────
            prev_best = fitness[alpha_idx]

            for i in range(WOLVES):
                if i in [alpha_idx, beta_idx, delta_idx]:
                    continue

                new_wolf = {}
                for key in ["lr", "dropout", "weight_decay"]:
                    r1, r2 = np.random.rand(), np.random.rand()

                    def step_from(leader):
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        D = abs(C * leader[key] - wolves[i][key])
                        return leader[key] - A * D

                    X_alpha = step_from(alpha)
                    X_beta  = step_from(beta)
                    X_delta = step_from(delta)

                    new_wolf[key] = (
                        weights["alpha"] * X_alpha +
                        weights["beta"]  * X_beta  +
                        weights["delta"] * X_delta
                    )

                new_wolf["lr"]           = float(np.clip(new_wolf["lr"],           1e-5, 1e-3))
                new_wolf["dropout"]      = float(np.clip(new_wolf["dropout"],      0.1,  0.5))
                new_wolf["weight_decay"] = float(np.clip(new_wolf["weight_decay"], 1e-5, 1e-3))
                new_wolf["batch_size"]   = float(np.random.choice([16, 32]))

                new_fit = proxy_train(new_wolf, train_dataset, val_dataset)
                if new_fit > fitness[i]:
                    wolves[i]  = new_wolf
                    fitness[i] = new_fit

            new_best = max(fitness)
            print(f"  Best acc: {new_best:.4f} | "
                  f"lr={wolves[int(np.argmax(fitness))]['lr']:.2e} | "
                  f"dropout={wolves[int(np.argmax(fitness))]['dropout']:.2f}")

        # ── REINFORCE update for RL meta-controller ─────────────────
        # Reward for the entire episode = final best fitness
        episode_reward = max(fitness)
        rl_baseline    = 0.9 * rl_baseline + 0.1 * episode_reward
        advantage      = episode_reward - rl_baseline

        policy_loss = -sum(lp * advantage for _, _, lp in trajectory) / len(trajectory)
        meta_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_ctrl.parameters(), max_norm=1.0)
        meta_opt.step()

        print(f"\n  RL Episode {rl_ep} | episode_reward={episode_reward:.4f} | "
              f"advantage={advantage:.4f} | policy_loss={policy_loss.item():.4f}")

        # ── Track global best across episodes ───────────────────────
        ep_best_idx = int(np.argmax(fitness))
        if fitness[ep_best_idx] > best_acc_overall:
            best_acc_overall = fitness[ep_best_idx]
            best_hp_overall  = copy.deepcopy(wolves[ep_best_idx])

    # Save RL meta-controller weights
    ctrl_path = os.path.join(SAVE_DIR, "rl_meta_controller.pth")
    torch.save(meta_ctrl.state_dict(), ctrl_path)
    print(f"\n  💾 RL meta-controller saved → {ctrl_path}")

    print("\n" + "=" * 60)
    print("✅ RL-Guided φ-MGWO Finished — Best Hyperparameters")
    print(f"  lr           : {best_hp_overall['lr']:.6f}")
    print(f"  dropout      : {best_hp_overall['dropout']:.4f}")
    print(f"  weight_decay : {best_hp_overall['weight_decay']:.6f}")
    print(f"  batch_size   : {int(best_hp_overall['batch_size'])}")
    print(f"  proxy val acc: {best_acc_overall:.4f}")
    print("=" * 60)

    return best_hp_overall


# ─────────────────────────────────────────
# FULL TRAINING + TEST EVAL
# ─────────────────────────────────────────
def full_train_and_eval(best_hp, train_dataset, val_dataset, test_dataset):
    print("\n" + "=" * 60)
    print("Full Training with Best Hyperparameters (from RL-MGWO)")
    print("=" * 60)

    batch_size = int(best_hp["batch_size"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_swin_t(NUM_CLASSES, dropout=best_hp["dropout"])

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params,     "lr": best_hp["lr"]},
            {"params": backbone_params, "lr": best_hp["lr"] * 0.1},
        ],
        weight_decay=best_hp["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FULL_EPOCHS
    )

    best_val_acc   = 0.0
    best_state     = None
    patience_count = 0

    for epoch in range(1, FULL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{FULL_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= EARLY_PATIENCE:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    best_model_path = os.path.join(SAVE_DIR, "swin_t_rl_mgwo_best.pth")
    torch.save(best_state, best_model_path)
    print(f"\n💾 Best model saved to: {best_model_path}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 60)
    print("Test Set Performance")
    print("=" * 60)
    print(f"  Test Loss   : {test_loss:.4f}")
    print(f"  Test Acc    : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1     : {test_f1:.4f}")
    print("\nPer-class Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix (Test Set) — RL-Guided φ-MGWO + Swin-T")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_test.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # ROC curves
    try:
        auc_macro = roc_auc_score(test_labels, test_probs, multi_class="ovr", average="macro")
        print(f"\nTest ROC-AUC (macro): {auc_macro:.4f}")

        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
        plt.figure(figsize=(8, 6))
        for i, (cls, color) in enumerate(zip(CLASSES, colors)):
            y_true = (test_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true, test_probs[:, i])
            auc_cls = roc_auc_score(y_true, test_probs[:, i])
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc_cls:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (Test Set) — RL-Guided φ-MGWO + Swin-T")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(RESULTS_DIR, "roc_curves_test.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curves saved to: {roc_path}")
    except Exception as e:
        print(f"ROC-AUC calculation failed: {e}")

    return test_acc, test_f1


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading datasets...")
    train_dataset = MangoDataset(os.path.join(SPLITS_DIR, "train"), CLASSES, transform=train_transforms)
    val_dataset   = MangoDataset(os.path.join(SPLITS_DIR, "val"),   CLASSES, transform=val_test_transforms)
    test_dataset  = MangoDataset(os.path.join(SPLITS_DIR, "test"),  CLASSES, transform=val_test_transforms)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val  : {len(val_dataset)} images")
    print(f"  Test : {len(test_dataset)} images")

    # RL-guided φ-MGWO hyperparameter search
    # rl_meta_episodes: how many full GWO runs the RL agent learns from
    # increase for better policy (3 is a good starting point)
    best_hp = rl_gwo_search(train_dataset, val_dataset, rl_meta_episodes=3)

    # Full Swin-T training with best hyperparameters
    test_acc, test_f1 = full_train_and_eval(best_hp, train_dataset, val_dataset, test_dataset)

    print("\nFinal Test Accuracy: {:.4f} ({:.2f}%)".format(test_acc, test_acc * 100))
    print("Final Test F1 Score: {:.4f}".format(test_f1))


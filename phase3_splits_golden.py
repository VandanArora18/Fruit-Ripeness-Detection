import os
import shutil
import random
import copy
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
CLAHE_DIR  = r"D:\RL\clahe_preprocessed"
SPLITS_DIR = r"D:\RL\splits"

CLASSES = [
    "Stage_0 (Unripe)",
    "Stage_1(Early Ripe)",
    "Stage_2 (Partially Ripe)",
    "Stage_3 (Ripe)",
]

# Ratios for split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# Golden ratio (kept for golden crop logic)
PHI = 1.618

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────
# CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(SPLITS_DIR, split, cls), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  AUGMENTATION OPERATIONS (action space for RL agent)
#  Each function takes an image (H×W×C uint8) and returns augmented.
# ═══════════════════════════════════════════════════════════════════

def aug_hflip(image, magnitude):
    """Horizontal flip — magnitude unused."""
    return cv2.flip(image, 1)

def aug_vflip(image, magnitude):
    """Vertical flip — magnitude unused."""
    return cv2.flip(image, 0)

def aug_rotate(image, magnitude):
    """Rotate by angle = magnitude * 30 degrees (range -30 to +30)."""
    h, w = image.shape[:2]
    angle  = (magnitude - 0.5) * 60  # maps [0,1] → [-30, 30]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_brightness(image, magnitude):
    """Brightness shift: magnitude in [0,1] → beta in [-40, 40]."""
    beta = int((magnitude - 0.5) * 80)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

def aug_contrast(image, magnitude):
    """Contrast scale: magnitude in [0,1] → alpha in [0.6, 1.4]."""
    alpha = 0.6 + magnitude * 0.8
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def aug_saturation(image, magnitude):
    """Saturation shift in HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    scale = 0.5 + magnitude * 1.0  # [0.5, 1.5]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_sharpen(image, magnitude):
    """Unsharp mask sharpening; magnitude controls strength."""
    strength = 0.5 + magnitude * 1.5  # [0.5, 2.0]
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kernel   = np.eye(3) + (kernel - np.eye(3)) * strength
    return np.clip(cv2.filter2D(image, -1, kernel), 0, 255).astype(np.uint8)

def aug_golden_tl(image, magnitude):
    """Golden ratio crop — top-left, resized back."""
    h, w = image.shape[:2]
    x = max(1, min(w-1, int(w / PHI)))
    y = max(1, min(h-1, int(h / PHI)))
    crop = image[0:y, 0:x]
    return cv2.resize(crop, (w, h)) if crop.size != 0 else image

def aug_golden_tr(image, magnitude):
    """Golden ratio crop — top-right, resized back."""
    h, w = image.shape[:2]
    x = max(1, min(w-1, int(w / PHI)))
    y = max(1, min(h-1, int(h / PHI)))
    crop = image[0:y, x:w]
    return cv2.resize(crop, (w, h)) if crop.size != 0 else image

def aug_golden_bl(image, magnitude):
    """Golden ratio crop — bottom-left, resized back."""
    h, w = image.shape[:2]
    x = max(1, min(w-1, int(w / PHI)))
    y = max(1, min(h-1, int(h / PHI)))
    crop = image[y:h, 0:x]
    return cv2.resize(crop, (w, h)) if crop.size != 0 else image

def aug_golden_br(image, magnitude):
    """Golden ratio crop — bottom-right, resized back."""
    h, w = image.shape[:2]
    x = max(1, min(w-1, int(w / PHI)))
    y = max(1, min(h-1, int(h / PHI)))
    crop = image[y:h, x:w]
    return cv2.resize(crop, (w, h)) if crop.size != 0 else image

# Full action space: list of (name, function)
AUGMENTATION_POOL = [
    ("hflip",      aug_hflip),
    ("vflip",      aug_vflip),
    ("rotate",     aug_rotate),
    ("brightness", aug_brightness),
    ("contrast",   aug_contrast),
    ("saturation", aug_saturation),
    ("sharpen",    aug_sharpen),
    ("golden_tl",  aug_golden_tl),
    ("golden_tr",  aug_golden_tr),
    ("golden_bl",  aug_golden_bl),
    ("golden_br",  aug_golden_br),
]

NUM_AUGMENTATIONS = len(AUGMENTATION_POOL)   # 11


# ═══════════════════════════════════════════════════════════════════
#  RL AUGMENTATION POLICY AGENT
#
#  Treats augmentation selection as a bandit / policy-gradient problem.
#
#  State  : one-hot class label (4-dim)  +  image statistics (5-dim)
#           = 9-dim state vector
#  Action : pick K augmentation operations from AUGMENTATION_POOL
#           (each action also has a continuous magnitude in [0,1])
#  Reward : proxy validation accuracy of a lightweight CNN trained on
#           augmented images vs. original images
#
#  Policy : small MLP (9 → 64 → NUM_AUGMENTATIONS probabilities)
#           trained with REINFORCE (policy gradient)
# ═══════════════════════════════════════════════════════════════════

class AugPolicyNet(nn.Module):
    """
    Lightweight policy network.
    Input : state vector (class one-hot + image stats) → 9-dim
    Output: probability distribution over NUM_AUGMENTATIONS operations
    """
    def __init__(self, state_dim=9, hidden=64, num_actions=NUM_AUGMENTATIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def image_stats(img):
    """
    Compute 5 simple statistics from a BGR uint8 image.
    Returns a numpy array of shape (5,), values normalised to [0,1].
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean    = gray.mean()   / 255.0
    std     = gray.std()    / 128.0
    # Laplacian variance → proxy for sharpness, clamp to [0,1]
    lap_var = np.var(cv2.Laplacian(gray, cv2.CV_32F)) / 10000.0
    lap_var = np.clip(lap_var, 0.0, 1.0)
    # Colour saturation mean
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_mean = hsv[:, :, 1].mean() / 255.0
    # Brightness (V channel)
    bri_mean = hsv[:, :, 2].mean() / 255.0
    return np.array([mean, std, lap_var, sat_mean, bri_mean], dtype=np.float32)


def build_state(class_idx, img):
    """Build 9-dim state: 4-dim one-hot class + 5-dim image stats."""
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[class_idx] = 1.0
    stats = image_stats(img)
    return np.concatenate([one_hot, stats])   # shape (9,)


# ─── Proxy classifier (tiny CNN) used to score augmentation quality ───

class TinyCNN(nn.Module):
    """Fast 3-conv proxy classifier for RL reward computation."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),  # 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_proxy_loaders(images_by_class, aug_ops, batch_size=16):
    """
    Build train/val DataLoaders for the proxy CNN using the
    given aug_ops list (each entry: (aug_fn, magnitude)).
    Returns train_loader, val_loader.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),   # small for speed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    train_imgs, train_lbs, val_imgs, val_lbs = [], [], [], []

    for cls_idx, (cls_name, img_list) in enumerate(images_by_class.items()):
        for i, img in enumerate(img_list):
            # original always goes to train
            train_imgs.append(img)
            train_lbs.append(cls_idx)
            # augmented versions
            for aug_fn, mag in aug_ops:
                aug_img = aug_fn(img.copy(), mag)
                train_imgs.append(aug_img)
                train_lbs.append(cls_idx)

        # last 20% of originals go to val (no augmentation)
        n_val = max(1, len(img_list) // 5)
        for img in img_list[-n_val:]:
            val_imgs.append(img)
            val_lbs.append(cls_idx)

    class _DS(Dataset):
        def __init__(self, imgs, lbs, tf):
            self.imgs = imgs
            self.lbs  = lbs
            self.tf   = tf

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img = cv2.cvtColor(self.imgs[idx], cv2.COLOR_BGR2RGB)
            x   = self.tf(img)
            return x, torch.tensor(self.lbs[idx], dtype=torch.long)

    tr_ds = _DS(train_imgs, train_lbs, transform)
    va_ds = _DS(val_imgs,   val_lbs,   transform)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return tr_ld, va_ld


def proxy_accuracy(train_loader, val_loader, epochs=4, device="cpu"):
    """Train TinyCNN and return validation accuracy."""
    model = TinyCNN(num_classes=4).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds   = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    return correct / total if total > 0 else 0.0


# ─── REINFORCE training loop ───

def train_rl_aug_policy(
    images_by_class,          # dict: cls_name → list of BGR uint8 ndarrays
    rl_episodes   = 25,       # number of policy-gradient episodes
    augs_per_img  = 3,        # how many augmentations to apply per image
    proxy_epochs  = 4,        # proxy CNN training epochs per episode
    device        = "cpu",
):
    """
    Train the RL augmentation policy using REINFORCE.

    Returns:
        policy_net  : trained AugPolicyNet (saved weights)
        best_ops    : list of (aug_fn, magnitude) — best policy found
    """
    print("\n" + "=" * 70)
    print("RL AUGMENTATION POLICY TRAINING (REINFORCE)")
    print(f"  Episodes      : {rl_episodes}")
    print(f"  Augs per img  : {augs_per_img}")
    print(f"  Proxy epochs  : {proxy_epochs}")
    print("=" * 70)

    policy_net = AugPolicyNet().to(device)
    optimizer  = torch.optim.Adam(policy_net.parameters(), lr=5e-3)

    # Baseline (running mean reward for variance reduction)
    baseline   = 0.0
    alpha_bl   = 0.9           # EMA factor for baseline

    best_reward    = -1.0
    best_ops       = []
    best_policy_sd = copy.deepcopy(policy_net.state_dict())

    # Pre-load a few representative images per class for state computation
    sample_imgs = {cls: imgs[:3] for cls, imgs in images_by_class.items()}

    for episode in range(1, rl_episodes + 1):
        policy_net.train()

        # ── Collect actions for each class ──────────────────────────
        log_probs_ep = []
        chosen_ops   = []   # (aug_fn, magnitude) pairs for this episode

        for cls_idx, (cls_name, img_list) in enumerate(images_by_class.items()):
            rep_img  = img_list[0]
            state_np = build_state(cls_idx, rep_img)
            state_t  = torch.tensor(state_np, dtype=torch.float32).to(device)

            probs = policy_net(state_t)   # (NUM_AUGMENTATIONS,)

            # Sample augs_per_img operations (with replacement)
            dist = torch.distributions.Categorical(probs)
            for _ in range(augs_per_img):
                action = dist.sample()
                log_probs_ep.append(dist.log_prob(action))
                aug_fn   = AUGMENTATION_POOL[action.item()][1]
                magnitude = float(torch.rand(1).item())
                chosen_ops.append((aug_fn, magnitude))

        # ── Compute reward via proxy CNN ─────────────────────────────
        tr_ld, va_ld = build_proxy_loaders(images_by_class, chosen_ops)
        reward = proxy_accuracy(tr_ld, va_ld, epochs=proxy_epochs, device=device)

        # ── Update baseline ──────────────────────────────────────────
        baseline = alpha_bl * baseline + (1 - alpha_bl) * reward
        advantage = reward - baseline

        # ── REINFORCE gradient update ────────────────────────────────
        policy_loss = -sum(lp * advantage for lp in log_probs_ep) / len(log_probs_ep)
        optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Track best policy ────────────────────────────────────────
        if reward > best_reward:
            best_reward    = reward
            best_ops       = list(chosen_ops)
            best_policy_sd = copy.deepcopy(policy_net.state_dict())

        # ── Logging ─────────────────────────────────────────────────
        aug_names = [AUGMENTATION_POOL[i][0]
                     for i in range(len(AUGMENTATION_POOL))
                     if any(AUGMENTATION_POOL[i][1] == fn for fn, _ in chosen_ops)]
        print(f"  Ep {episode:03d}/{rl_episodes} | Reward(val_acc): {reward:.4f} | "
              f"Baseline: {baseline:.4f} | Loss: {policy_loss.item():.4f}")

    policy_net.load_state_dict(best_policy_sd)
    print(f"\n✅ RL Policy Training Done. Best proxy val_acc = {best_reward:.4f}")

    # Save policy weights
    policy_path = os.path.join(
        os.path.dirname(SPLITS_DIR), "rl_aug_policy.pth"
    )
    torch.save(best_policy_sd, policy_path)
    print(f"   Policy saved → {policy_path}")

    return policy_net, best_ops


def apply_rl_policy_to_image(policy_net, class_idx, img, augs_per_img=3, device="cpu"):
    """
    Given a trained policy, deterministically select the top-k augmentations
    (highest probability) for the given class+image, apply them and return
    a list of augmented images (each 224×224).
    """
    policy_net.eval()
    state_np = build_state(class_idx, img)
    state_t  = torch.tensor(state_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = policy_net(state_t)   # (NUM_AUGMENTATIONS,)

    top_k_indices = torch.topk(probs, k=augs_per_img).indices.tolist()

    augmented = []
    for idx in top_k_indices:
        aug_name, aug_fn = AUGMENTATION_POOL[idx]
        magnitude = float(probs[idx].item())   # use prob as magnitude proxy
        aug_img   = aug_fn(img.copy(), magnitude)
        aug_img   = cv2.resize(aug_img, (224, 224))
        augmented.append((aug_name, aug_img))

    return augmented


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — BUILD SPLIT LISTS  (70 / 10 / 20)
#  ⚠ This is where train / val / test split happens ⚠
# ═══════════════════════════════════════════════════════════════════
def build_splits():
    """
    Stratified split of CLAHE-preprocessed images into train/val/test.

    SPLIT RATIOS: 70% Train  |  10% Val  |  20% Test
    Random state: 42 (reproducible)

    Returns split_data dict and images_by_class dict (for RL training).
    """
    print("\n" + "=" * 70)
    print("STEP 1 — Creating 70/10/20 stratified split from CLAHE-preprocessed data")
    print("=" * 70)

    split_data       = {"train": [], "val": [], "test": []}
    images_by_class  = {}   # cls_name → list of BGR ndarrays (train only)

    for cls in CLASSES:
        cls_dir = os.path.join(CLAHE_DIR, cls)
        files   = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # ── First split: (train+val) vs test ──────────────────────────
        train_val, test = train_test_split(
            files,
            test_size=TEST_RATIO,
            random_state=42
        )

        # ── Second split: train vs val ────────────────────────────────
        val_size_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train, val   = train_test_split(
            train_val,
            test_size=val_size_adj,
            random_state=42
        )

        print(f"\n  📂 {cls}")
        print(f"     Total : {len(files)}")
        print(f"     Train : {len(train)}")
        print(f"     Val   : {len(val)}")
        print(f"     Test  : {len(test)}")

        split_data["train"].append((cls, train))
        split_data["val"].append((cls, val))
        split_data["test"].append((cls, test))

        # Load train images into memory for RL policy training
        imgs_loaded = []
        for fname in train:
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                imgs_loaded.append(img)
        images_by_class[cls] = imgs_loaded

    return split_data, images_by_class


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — COPY VAL & TEST (NO AUGMENTATION)
# ═══════════════════════════════════════════════════════════════════
def copy_val_test(split_data):
    print("\n" + "=" * 70)
    print("STEP 2 — Copying validation and test sets (no augmentation)")
    print("=" * 70)

    for split in ["val", "test"]:
        total = 0
        for cls, files in split_data[split]:
            src_dir = os.path.join(CLAHE_DIR, cls)
            dst_dir = os.path.join(SPLITS_DIR, split, cls)

            for fname in files:
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(src, dst)
                total += 1

        print(f"  ✅ {split.upper()} — {total} images copied.")


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN SET: ORIGINAL + RL-LEARNED AUGMENTATIONS
#            (replaces the fixed golden-ratio + flip + rotate)
# ═══════════════════════════════════════════════════════════════════
def build_train_with_rl_policy(split_data, policy_net, augs_per_img=3, device="cpu"):
    """
    For every training image:
      1. Save the original
      2. Query the RL policy to get the top-k augmentations for
         (class, image) pair and save each augmented version.

    Net result: (1 + augs_per_img) images saved per original.
    """
    print("\n" + "=" * 70)
    print("STEP 3 — Building TRAIN set with RL-learned augmentation policy")
    print("=" * 70)

    for cls_idx, (cls, files) in enumerate(split_data["train"]):
        src_dir = os.path.join(CLAHE_DIR, cls)
        dst_dir = os.path.join(SPLITS_DIR, "train", cls)

        saved = 0

        for fname in files:
            img_path = os.path.join(src_dir, fname)
            img      = cv2.imread(img_path)

            if img is None:
                continue

            img  = cv2.resize(img, (224, 224))
            stem = Path(fname).stem

            # 1) Save original
            cv2.imwrite(os.path.join(dst_dir, f"{stem}_orig.jpg"), img)
            saved += 1

            # 2) RL policy augmentations
            aug_list = apply_rl_policy_to_image(
                policy_net, cls_idx, img,
                augs_per_img=augs_per_img,
                device=device
            )
            for i, (aug_name, aug_img) in enumerate(aug_list):
                out_path = os.path.join(dst_dir, f"{stem}_rl_{aug_name}_{i+1}.jpg")
                cv2.imwrite(out_path, aug_img)
                saved += 1

        print(f"  📂 {cls}  →  {saved} train images (orig + {augs_per_img} RL augs each)")


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — VERIFY FINAL SPLITS
# ═══════════════════════════════════════════════════════════════════
def verify_splits():
    print("\n📁 Final split structure (after RL augmentation policy):")
    print("=" * 70)

    grand_total = 0
    for split in ["train", "val", "test"]:
        split_total = 0
        print(f"\n  📂 {split}/")
        for cls in CLASSES:
            cls_dir = os.path.join(SPLITS_DIR, split, cls)
            count   = len([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"       📂 {cls}/  →  {count} images")
            split_total += count
        print(f"     Subtotal: {split_total}")
        grand_total += split_total

    print("=" * 70)
    print(f"  ✅ Grand Total (all splits): {grand_total} images")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {DEVICE}\n")

    # STEP 1 — Split data  (70 train / 10 val / 20 test)
    splits, images_by_class = build_splits()

    # STEP 2 — Copy val + test as-is
    copy_val_test(splits)

    # STEP 3a — Train RL augmentation policy
    policy_net, best_ops = train_rl_aug_policy(
        images_by_class,
        rl_episodes  = 25,    # increase for better policy (costs more time)
        augs_per_img = 3,     # number of augmentations per training image
        proxy_epochs = 4,     # proxy CNN epochs per episode
        device       = DEVICE,
    )

    # STEP 3b — Apply RL policy to build train set
    build_train_with_rl_policy(splits, policy_net, augs_per_img=3, device=DEVICE)

    # STEP 4 — Verify
    verify_splits()

    print("\n✅ Phase 3 (RL Augmentation Policy) Complete!")
    print(f"   Splits saved in : {SPLITS_DIR}")


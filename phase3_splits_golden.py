import os, shutil, random, cv2, torch
from pathlib import Path
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────── CONFIG & PATHS ───────────────────
CLAHE_DIR  = r"D:\RL\clahe_preprocessed"
SPLITS_DIR = r"D:\RL\splits"
CLASSES    = ["Stage_0 (Unripe)", "Stage_1(Early Ripe)", "Stage_2 (Partially Ripe)", "Stage_3 (Ripe)"]
PHI, SEED  = 1.618, 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(SPLITS_DIR, split, cls), exist_ok=True)

# ─────────────────── AUGMENTATION POOL (11 ACTIONS) ───────────────────
def golden_crop(img, idx):
    h, w = img.shape[:2]
    x, y = int(w / PHI), int(h / PHI)
    quads = [img[:y, :x], img[:y, x:], img[y:, :x], img[y:, x:]]
    c = quads[idx]
    return cv2.resize(c, (w, h)) if c.size > 0 else img

def rotate_img(img, mag):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), (mag - 0.5) * 60, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def color_jitter(img, mode, mag):
    if mode == "bri": return cv2.convertScaleAbs(img, alpha=1.0, beta=int((mag - 0.5) * 80))
    if mode == "con": return cv2.convertScaleAbs(img, alpha=0.6 + mag * 0.8, beta=0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.5 + mag), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

AUG_POOL = [
    ("hflip",       lambda img, m: cv2.flip(img, 1)),
    ("vflip",       lambda img, m: cv2.flip(img, 0)),
    ("rotate",      rotate_img),
    ("brightness",  lambda img, m: color_jitter(img, "bri", m)),
    ("contrast",    lambda img, m: color_jitter(img, "con", m)),
    ("saturation",  lambda img, m: color_jitter(img, "sat", m)),
    ("sharpen",     lambda img, m: cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))),
    ("golden_tl",   lambda img, m: golden_crop(img, 0)),
    ("golden_tr",   lambda img, m: golden_crop(img, 1)),
    ("golden_bl",   lambda img, m: golden_crop(img, 2)),
    ("golden_br",   lambda img, m: golden_crop(img, 3)),
]

# ─────────────────── RL POLICY NETWORK & STATE ───────────────────
class AugPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, len(AUG_POOL)))
    def forward(self, x): return torch.softmax(self.net(x), dim=-1)

def get_state(class_idx, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    one_hot = np.zeros(4, dtype=np.float32); one_hot[class_idx] = 1.0
    stats = np.array([
        gray.mean() / 255.0, gray.std() / 128.0,
        np.clip(np.var(cv2.Laplacian(gray, cv2.CV_32F)) / 10000.0, 0, 1),
        hsv[:, :, 1].mean() / 255.0, hsv[:, :, 2].mean() / 255.0
    ], dtype=np.float32)
    return np.concatenate([one_hot, stats])

# ─────────────────── PROXY CNN FOR RL REWARD ───────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Linear(32 * 16, 4)
    def forward(self, x): return self.fc(self.feat(x).flatten(1))

class FastDataset(Dataset):
    def __init__(self, imgs, labels): self.imgs, self.labels = imgs, labels
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        x = cv2.resize(cv2.cvtColor(self.imgs[idx], cv2.COLOR_BGR2RGB), (64, 64))
        return torch.from_numpy(x).permute(2, 0, 1).float() / 255.0, self.labels[idx]

def eval_proxy_reward(images_by_class, ops, device="cpu"):
    train_x, train_y, val_x, val_y = [], [], [], []
    for c_idx, imgs in images_by_class.items():
        n_val = max(1, len(imgs) // 5)
        for img in imgs[:-n_val]:
            train_x.append(img); train_y.append(c_idx)
            for fn, mag in ops:
                train_x.append(fn(img.copy(), mag)); train_y.append(c_idx)
        for img in imgs[-n_val:]: val_x.append(img); val_y.append(c_idx)

    tr_ld = DataLoader(FastDataset(train_x, train_y), batch_size=32, shuffle=True)
    va_ld = DataLoader(FastDataset(val_x, val_y), batch_size=32, shuffle=False)

    m = TinyCNN().to(device)
    opt, crit = torch.optim.Adam(m.parameters(), lr=1e-3), nn.CrossEntropyLoss()
    for _ in range(3):
        m.train()
        for x, y in tr_ld: opt.zero_grad(); crit(m(x.to(device)), y.to(device)).backward(); opt.step()
    
    m.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in va_ld:
            correct += (m(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / max(total, 1)

# ─────────────────── REINFORCE TRAINING ───────────────────
def train_rl_policy(images_by_class, episodes=15, device="cpu"):
    print("\n" + "=" * 50 + "\nTraining RL Augmentation Policy (REINFORCE)\n" + "=" * 50)
    net, opt = AugPolicyNet().to(device), torch.optim.Adam(AugPolicyNet().parameters(), lr=5e-3)
    baseline, best_reward, best_sd = 0.0, -1.0, None

    for ep in range(1, episodes + 1):
        net.train(); log_probs, ops = [], []
        for c_idx, imgs in images_by_class.items():
            st = torch.tensor(get_state(c_idx, imgs[0]), dtype=torch.float32).to(device)
            probs = net(st)
            dist = torch.distributions.Categorical(probs)
            for _ in range(3):
                act = dist.sample()
                log_probs.append(dist.log_prob(act))
                ops.append((AUG_POOL[act.item()][1], float(torch.rand(1).item())))

        reward = eval_proxy_reward(images_by_class, ops, device)
        baseline = 0.9 * baseline + 0.1 * reward
        loss = -sum(lp * (reward - baseline) for lp in log_probs) / len(log_probs)

        opt.zero_grad(); loss.backward(); opt.step()
        if reward > best_reward: best_reward, best_sd = reward, net.state_dict()
        print(f"  Episode {ep:02d}/{episodes} | Proxy Val Acc (Reward): {reward:.4f} | Loss: {loss.item():.4f}")

    net.load_state_dict(best_sd)
    torch.save(best_sd, r"D:\RL\rl_aug_policy.pth")
    print(f"✅ Policy trained & saved. Best Proxy Acc: {best_reward:.4f}")
    return net

# ─────────────────── DATASET SPLITTING & SAVING ───────────────────
def run_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 1: Stratified split (70% train, 10% val, 20% test)
    splits, images_by_class = {"train": [], "val": [], "test": []}, {}
    for c_idx, cls in enumerate(CLASSES):
        files = sorted([f for f in os.listdir(os.path.join(CLAHE_DIR, cls)) if f.lower().endswith(('.jpg','.png'))])
        tr_val, test = train_test_split(files, test_size=TEST_RATIO, random_state=SEED)
        train, val   = train_test_split(tr_val, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), random_state=SEED)
        
        splits["train"].append((cls, train)); splits["val"].append((cls, val)); splits["test"].append((cls, test))
        images_by_class[c_idx] = [cv2.resize(cv2.imread(os.path.join(CLAHE_DIR, cls, f)), (224, 224)) for f in train]

    # Step 2: Copy val & test without modification
    for sp in ["val", "test"]:
        for cls, files in splits[sp]:
            for f in files: shutil.copy2(os.path.join(CLAHE_DIR, cls, f), os.path.join(SPLITS_DIR, sp, cls, f))
    print("✅ Clean Validation and Test sets copied.")

    # Step 3: Train RL policy
    policy = train_rl_policy(images_by_class, episodes=15, device=device)
    policy.eval()

    # Step 4: Augment Train Set with RL Policy
    print("\nGenerating RL-Augmented Training Set...")
    for c_idx, (cls, files) in enumerate(splits["train"]):
        dst_dir = os.path.join(SPLITS_DIR, "train", cls)
        for f in files:
            img = cv2.resize(cv2.imread(os.path.join(CLAHE_DIR, cls, f)), (224, 224))
            stem = Path(f).stem
            cv2.imwrite(os.path.join(dst_dir, f"{stem}_orig.jpg"), img)

            # Query policy for top-3 augmentations for this image
            st = torch.tensor(get_state(c_idx, img), dtype=torch.float32).to(device)
            with torch.no_grad(): probs = policy(st)
            top3 = torch.topk(probs, k=3).indices.tolist()

            for i, act_idx in enumerate(top3):
                aug_name, fn = AUG_POOL[act_idx]
                aug_img = fn(img.copy(), float(probs[act_idx].item()))
                cv2.imwrite(os.path.join(dst_dir, f"{stem}_rl_{aug_name}_{i+1}.jpg"), cv2.resize(aug_img, (224, 224)))
        print(f"  {cls} augmented successfully (4x expansion).")

    # Final summary count
    print("\n" + "=" * 50 + "\nFinal Dataset Summary:")
    for sp in ["train", "val", "test"]:
        cnt = sum(len(os.listdir(os.path.join(SPLITS_DIR, sp, c))) for c in CLASSES)
        print(f"  {sp.upper()}: {cnt} images")
    print("=" * 50 + "\nPhase 3 Complete!")

if __name__ == "__main__":
    run_pipeline()

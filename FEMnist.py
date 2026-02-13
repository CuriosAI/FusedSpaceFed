
# ============
# 0) Imports & Setup
# ============
import os, math, random, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score

SEED = 41
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from torch.amp import autocast, GradScaler
SCALER = GradScaler("cuda") if device.type == "cuda" else None


# Percorsi ai file .npy FEMNIST salvati in Drive
TRAIN_X_PATH = "/content/drive/MyDrive/Colab Notebooks/FedLearning/femnist_npy/X_train_femnist.npy"
TRAIN_Y_PATH = "/content/drive/MyDrive/Colab Notebooks/FedLearning/femnist_npy/y_train_femnist.npy"
TEST_X_PATH  = "/content/drive/MyDrive/Colab Notebooks/FedLearning/femnist_npy/X_test_femnist.npy"
TEST_Y_PATH  = "/content/drive/MyDrive/Colab Notebooks/FedLearning/femnist_npy/y_test_femnist.npy"

def load_femnist_numpy():
    X_train = np.load(TRAIN_X_PATH)                      # (N, 28, 28, 1), float32 in [0,1]
    y_train = np.load(TRAIN_Y_PATH).astype(np.int64).reshape(-1)  # 0..61
    X_test  = np.load(TEST_X_PATH)
    y_test  = np.load(TEST_Y_PATH).astype(np.int64).reshape(-1)
    return X_train, y_train, X_test, y_test

class RetinaNPYDataset(Dataset):
    """
    Dataset semplice che prende X (N,128,128,3) e y (N,) numpy
    e restituisce (tensor[3,H,W] in [0,1], label long).
    """
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]        # (H,W) o (H,W,1) per FEMNIST
        lbl = int(self.y[idx])   # 0..61

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # img può essere (H, W) o (H, W, 1) o (1, H, W)
        if img.ndim == 2:
            # (H,W) → (1,H,W)
            img = img.unsqueeze(0)

        elif img.ndim == 3:
            if img.shape[0] == 1 and img.shape[1] == 28 and img.shape[2] == 28:
                # già (1,H,W)
                pass
            elif img.shape[-1] == 1:
                # (H,W,1) → (1,H,W)
                img = img.permute(2, 0, 1)
            elif img.shape[-1] == 3:
                # (H,W,3) → (3,H,W) [non dovrebbe succedere con FEMNIST]
                img = img.permute(2, 0, 1)

        # Resize qualsiasi dimensione → (C,32,32)
        if img.shape[1] != 32 or img.shape[2] != 32:
            img = img.unsqueeze(0)  # (1,C,H,W)
            img = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)
            img = img.squeeze(0)    # (C,32,32)

        # ATTENZIONE: X_femnist.npy è già in [0,1]
        img = img.float()

        return img, lbl

class ExtremeNonIIDRetina(Dataset):
    """
    strategy: "dirichlet" o "pathological"
    """
    def __init__(self, X, y, train=True, strategy="dirichlet",
                 alpha=0.1, num_clients=10, classes_per_client=2):
        self.strategy = strategy
        self.alpha = alpha
        self.num_clients = num_clients
        self.classes_per_client = classes_per_client

        self.targets = torch.tensor(y, dtype=torch.long)
        self.num_classes = int(self.targets.max().item()) + 1

        self.base_dataset = RetinaNPYDataset(X, y)
        self.client_indices = self._create_partitions()

    def _create_partitions(self):
        if self.strategy == "dirichlet":
            return self._dirichlet_partition(self.targets, self.num_classes)
        elif self.strategy == "pathological":
            return self._pathological_partition(self.targets, self.num_classes)
        else:
            raise ValueError("Unknown strategy")

    def _dirichlet_partition(self, targets, num_classes):
        """
        Dirichlet con dimensione client quasi-costante.
        Ogni client riceve ~N/K esempi totali.
        """
        rng = np.random.default_rng(SEED)
        N = len(targets)
        K = self.num_clients

        # --- 1. target per client: quasi uguali ---
        base = N // K
        remainder = N % K
        client_sizes = np.array([base + (1 if i < remainder else 0) for i in range(K)])

        # --- 2. per ogni client, campiona un "preference" dirichlet sulle classi ---
        cls_pref = rng.dirichlet([self.alpha] * num_classes, size=K)  # shape K × C

        # --- 3. normalizza per ottenere distribuzioni valide ---
        cls_pref = cls_pref / cls_pref.sum(axis=1, keepdims=True)

        # --- 4. ordina gli indici per classe ---
        class_to_idx = [np.where(targets.numpy() == c)[0] for c in range(num_classes)]
        for c in range(num_classes):
            rng.shuffle(class_to_idx[c])

        # --- 5. per ogni client, scegli esattamente client_sizes[cid] esempi ---
        client_indices = [[] for _ in range(K)]

        # trasformo in quantità intere
        allocations = []
        for cid in range(K):
            raw = cls_pref[cid] * client_sizes[cid]
            alloc = np.floor(raw).astype(int)
            diff = client_sizes[cid] - alloc.sum()
            if diff > 0:
                extra = rng.choice(num_classes, size=diff, replace=True)
                for c in extra:
                    alloc[c] += 1
            allocations.append(alloc)

        # --- 6. assegno i record rispettando le allocazioni ---
        ptr = [0] * num_classes  # pointer per ogni classe

        for cid in range(K):
          alloc = allocations[cid]
          for c in range(num_classes):
            take = alloc[c]
            cls_idx = class_to_idx[c]
            if ptr[c] + take > len(cls_idx):
                take = max(0, len(cls_idx) - ptr[c])
            chosen = cls_idx[ptr[c] : ptr[c] + take]
            ptr[c] += take
            client_indices[cid].extend(chosen.tolist())

          rng.shuffle(client_indices[cid])

        # --- 7. fallback se un client resta vuoto (non dovrebbe accadere) ---
        for cid in range(K):
            if len(client_indices[cid]) == 0:
                donor = int(np.argmax([len(ci) for ci in client_indices]))
                half = len(client_indices[donor]) // 2
                client_indices[cid] = client_indices[donor][:half]
                client_indices[donor] = client_indices[donor][half:]

        return client_indices

    def _pathological_partition(self, targets, num_classes):
        rng = np.random.default_rng(SEED)
        client_classes = [[] for _ in range(self.num_clients)]
        perm = rng.permutation(num_classes).tolist()
        p = 0
        for cid in range(self.num_clients):
            while len(client_classes[cid]) < self.classes_per_client:
                client_classes[cid].append(perm[p % num_classes])
                p += 1
            rng.shuffle(client_classes[cid])
        class_to_idx = {c: torch.where(targets == c)[0].numpy() for c in range(num_classes)}
        for c in range(num_classes):
            rng.shuffle(class_to_idx[c])
        client_indices = [[] for _ in range(self.num_clients)]
        for c in range(num_classes):
            holders = [cid for cid in range(self.num_clients) if c in client_classes[cid]]
            parts = np.array_split(class_to_idx[c], len(holders))
            for cid, part in zip(holders, parts):
                client_indices[cid].extend(part.tolist())
        for cid in range(self.num_clients):
            rng.shuffle(client_indices[cid])
        return client_indices

    def get_client_loader(self, client_id, batch_size=128, num_workers=0, shuffle=True):
        idxs = self.client_indices[client_id]
        subset = Subset(self.base_dataset, idxs)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=False)

# ============
# 2) Models — ResNet20-v2 and U-Net AE
# ============
class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        out = self.relu1(self.bn1(x))
        sc = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += sc
        return out

class ResNet_CIFAR_V2(nn.Module):
    """
    ResNet20-v2 stile CIFAR (conv1 stride=1), ma può lavorare anche su 128x128.
    """
    def __init__(self, depth=20, num_classes=62, in_channels=1):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        widths = [16, 32, 64]
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16,        widths[0], n, stride=1)
        self.layer2 = self._make_layer(widths[0], widths[1], n, stride=2)
        self.layer3 = self._make_layer(widths[1], widths[2], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(widths[2], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [PreActBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(PreActBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

class UNetSmallAE(nn.Module):
    def __init__(self, base=16, in_ch=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU(True)
        )
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(True)
        )
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.final = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        z  = self.bott(p2)
        u2 = self.up2(z); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.final(d1)
        return out, z

    def get_decoder_state_dict(self):
        keep = ("up", "dec", "final")
        return {k: v for k, v in self.state_dict().items() if any(s in k for s in keep)}
    def load_decoder_state_dict(self, decoder_state):
        sd = self.state_dict()
        for k, v in decoder_state.items():
            if k in sd:
                sd[k] = v
        self.load_state_dict(sd)

    # helper per encoder/decoder params (per warm-up / L2-SP)
    def encoder_modules(self):
        return nn.ModuleList([self.enc1, self.pool1, self.enc2, self.pool2, self.bott])
    def decoder_modules(self):
        return nn.ModuleList([self.up2, self.dec2, self.up1, self.dec1, self.final])
    def encoder_parameters(self):
        return (p for m in self.encoder_modules() for p in m.parameters())
    def decoder_parameters(self):
        return (p for m in self.decoder_modules() for p in m.parameters())

# ============
# 3) Clients
# ============
class FedAvgClient:
    def __init__(self, cid, loader, num_classes=62):
        self.id = cid
        self.loader = loader
        self.model = ResNet_CIFAR_V2(depth=20, num_classes=num_classes, in_channels=1).to(device)
        self.opt = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.crit = nn.CrossEntropyLoss()
        self.scaler = SCALER

    def train(self, epochs=1):
        self.model.train()
        losses = []
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                self.opt.zero_grad()
                if self.scaler:
                    with autocast("cuda"):
                        logits = self.model(x)
                        loss = self.crit(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss = self.crit(logits, y)
                    loss.backward()
                    self.opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda":
                with autocast("cuda"):
                    logits = self.model(x)
            else:
                logits = self.model(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        return correct / total if total > 0 else 0.0

    def get_parameters(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
    def set_parameters(self, params):
        self.model.load_state_dict({k: v.to(device, non_blocking=True) for k, v in params.items()})

class FusedSpaceFedClient:
    def __init__(self, cid, loader, fusion_mode="sum", num_classes=62,
                 warmup_encoder_epochs=0,
                 lambda_sp=0.0):
        assert fusion_mode in ("sum", "concat")
        self.id = cid
        self.loader = loader
        self.fusion_mode = fusion_mode
        self.warmup_encoder_epochs = int(warmup_encoder_epochs)
        self.lambda_sp = float(lambda_sp)

        # grayscale → 1 canale
        ae_in = 1

        # fused input channels
        if fusion_mode == "sum":
          clf_in = 1
        else:  # concat
          clf_in = 2

        self.ae  = UNetSmallAE(base=16, in_ch=ae_in).to(device)
        self.clf = ResNet_CIFAR_V2(depth=20, num_classes=num_classes, in_channels=clf_in).to(device)

        self.ae_opt  = optim.SGD(self.ae.parameters(),  lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.clf_opt = optim.SGD(self.clf.parameters(), lr=0.1,  momentum=0.9, weight_decay=5e-4)
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()
        self.scaler = SCALER

        # Stato round (warm-up)
        self._warmup_epochs_left = 0

        # L2-SP: snapshot pesi encoder
        self._enc_prefixes = ("enc1", "pool1", "enc2", "pool2", "bott")
        self.enc_init = {
            n: p.detach().clone()
            for n, p in self.ae.named_parameters()
            if n.startswith(self._enc_prefixes)
        } if self.lambda_sp > 0 else {}

    def _fuse(self, x, tx):
        return x + tx if self.fusion_mode == "sum" else torch.cat([x, tx], dim=1)

    def _set_trainability(self, enc=True, dec=True, clf=True):
        for p in self.ae.encoder_parameters():
            p.requires_grad = enc
        for p in self.ae.decoder_parameters():
            p.requires_grad = dec
        for p in self.clf.parameters():
            p.requires_grad = clf

    def on_round_start(self):
        if self.warmup_encoder_epochs > 0:
            self._warmup_epochs_left = self.warmup_encoder_epochs
            self._set_trainability(enc=True, dec=False, clf=False)
        else:
            self._warmup_epochs_left = 0
            self._set_trainability(enc=True, dec=True, clf=True)

    def _ae_forward(self, x):
        tx, z = self.ae(x)
        return tx, z

    def _l2sp_encoder_penalty(self):
        if self.lambda_sp <= 0:
            return None
        sp = 0.0
        for n, p in self.ae.named_parameters():
            if n in self.enc_init:
                sp = sp + (p - self.enc_init[n]).pow(2).sum()
        return sp

    def _ae_step_with_optional_l2sp(self, x):
        self.ae_opt.zero_grad(set_to_none=True)
        if self.scaler:
            with autocast("cuda"):
                tx, _ = self._ae_forward(x)
                rloss = self.mse(tx, x)
                sp = self._l2sp_encoder_penalty()
                loss = rloss + (self.lambda_sp * sp if sp is not None else 0.0)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.ae_opt)
            self.scaler.update()
        else:
            tx, _ = self._ae_forward(x)
            rloss = self.mse(tx, x)
            sp = self._l2sp_encoder_penalty()
            loss = rloss + (self.lambda_sp * sp if sp is not None else 0.0)
            loss.backward()
            self.ae_opt.step()
        return rloss.item()

    def train(self, epochs=1):
        self.ae.train()
        self.clf.train()
        R, C = [], []

        # Fase A: warm-up encoder (se attivo, solo a inizio round)
        if self._warmup_epochs_left > 0:
            for _ in range(self._warmup_epochs_left):
                for x, y in self.loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    r = self._ae_step_with_optional_l2sp(x)
                    R.append(r)
                    with torch.no_grad():
                        tx, _ = self._ae_forward(x)
                        fused = self._fuse(x, tx)
                        if device.type == "cuda":
                            with autocast("cuda"):
                                logits = self.clf(fused)
                        else:
                            logits = self.clf(fused)
                        C.append(self.ce(logits, y).item())
            self._set_trainability(enc=True, dec=True, clf=True)
            self._warmup_epochs_left = 0

        # Fase B: training standard per il round
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                # AE step
                R.append(self._ae_step_with_optional_l2sp(x))
                # Classifier step
                self.clf_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    tx, _ = self._ae_forward(x)
                fused = self._fuse(x, tx)
                if self.scaler:
                    with autocast("cuda"):
                        logits = self.clf(fused)
                        closs = self.ce(logits, y)
                    self.scaler.scale(closs).backward()
                    self.scaler.step(self.clf_opt)
                    self.scaler.update()
                else:
                    logits = self.clf(fused)
                    closs = self.ce(logits, y)
                    closs.backward()
                    self.clf_opt.step()
                C.append(closs.item())

        return float(np.mean(R)) if R else 0.0, float(np.mean(C)) if C else 0.0

    @torch.no_grad()
    def evaluate(self, loader):
        self.ae.eval()
        self.clf.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tx, _ = self._ae_forward(x)
            fused = self._fuse(x, tx)
            if device.type == "cuda":
                with autocast("cuda"):
                    logits = self.clf(fused)
            else:
                logits = self.clf(fused)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        return correct / total if total > 0 else 0.0

    def get_classifier_params(self):
        return {k: v.detach().cpu().clone() for k, v in self.clf.state_dict().items()}
    def set_classifier_params(self, params):
        self.clf.load_state_dict({k: v.to(device, non_blocking=True) for k, v in params.items()})
    def get_decoder_params(self):
        dec = self.ae.get_decoder_state_dict()
        return {k: v.detach().cpu().clone() for k, v in dec.items()}
    def set_decoder_params(self, params):
        self.ae.load_decoder_state_dict({k: v.to(device, non_blocking=True) for k, v in params.items()})


class FedProxClient(FedAvgClient):
    """
    FedProx = FedAvg + μ * ||w - w_global||²
    Con la stessa rete ResNet20-v2
    """
    def __init__(self, cid, loader, num_classes=62, mu=0.01):
        super().__init__(cid, loader, num_classes)
        self.mu = mu
        self.global_params = None  # snapshot da server

    def set_global_params(self, params):
        self.global_params = {k: v.clone().to(device) for k, v in params.items()}

    def train(self, epochs=1):
        self.model.train()
        losses = []

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                self.opt.zero_grad()

                logits = self.model(x)
                loss = self.crit(logits, y)

                # ---- Proximal term ----
                if self.global_params:
                    prox = 0.0
                    for (name, p) in self.model.named_parameters():
                        prox += (p - self.global_params[name]).pow(2).sum()
                    loss = loss + (self.mu / 2.0) * prox

                loss.backward()
                self.opt.step()
                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

class ScaffoldClient(FedAvgClient):
    """
    SCAFFOLD: usa vettori di controllo c_i (client) e c (server)
    """
    def __init__(self, cid, loader, num_classes=62):
        super().__init__(cid, loader, num_classes)
        self.ci = {k: torch.zeros_like(v, device=device) for k, v in self.model.state_dict().items()}
        self.c  = None  # server control variate

    def set_server_control(self, c_server):
        self.c = {k: v.to(device) for k, v in c_server.items()}

    def train(self, epochs=1):
        self.model.train()
        losses = []

        w_before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(device), y.to(device)
                self.opt.zero_grad()
                logits = self.model(x)
                loss = self.crit(logits, y)
                loss.backward()

                # ---- Correzione gradiente ----
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if name in self.ci:
                            p.grad += self.ci[name] - self.c[name]

                self.opt.step()
                losses.append(loss.item())

        # ---- Aggiornamento c_i al termine dell'update ----
        with torch.no_grad():
            w_after = self.model.state_dict()
            delta = {k: w_after[k] - w_before[k] for k in w_before}
            for k in self.ci:
                self.ci[k] = self.ci[k] - self.c[k] + (1.0 / (epochs * len(self.loader))) * delta[k]

        return float(np.mean(losses)) if losses else 0.0

# ============
# 4) Server
# ============
class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    @staticmethod
    def _avg_params(param_dicts):
        if not param_dicts:
            return {}
        avg = {}
        keys = param_dicts[0].keys()
        for k in keys:
            tensors = [d[k] for d in param_dicts]
            t0 = tensors[0]
            if t0.dtype in (torch.int64, torch.int32, torch.bool):
                avg[k] = t0
            else:
                avg[k] = torch.stack(tensors, dim=0).mean(0)
        return avg

    def aggregate_fedavg(self, clients):
        avg = self._avg_params([c.get_parameters() for c in clients])
        for c in clients:
            c.set_parameters(avg)

    def aggregate_fusedspace(self, clients):
        avg_clf = self._avg_params([c.get_classifier_params() for c in clients])
        avg_dec = self._avg_params([c.get_decoder_params()   for c in clients])
        for c in clients:
            c.set_classifier_params(avg_clf)
            c.set_decoder_params(avg_dec)

class ScaffoldServer(Server):
    def __init__(self, num_clients, model_example):
        super().__init__(num_clients)
        self.c = {k: torch.zeros_like(v) for k, v in model_example.state_dict().items()}

    def aggregate(self, clients):
        # media dei delta c_i
        new_c = {}
        for k in self.c:
            new_c[k] = torch.stack([c.ci[k].cpu() for c in clients], dim=0).mean(0)
        self.c = new_c

        # media dei pesi modello (come FedAvg)
        avg = self._avg_params([c.get_parameters() for c in clients])
        for c in clients:
            c.set_parameters(avg)
            c.set_server_control(self.c)

# ============
# 4.5) PER-CLASS ACCURACY FUNCTION
# ============

def per_class_accuracy(model_or_client, loader, num_classes=62, fused=False):
    """
    Calcola l'accuracy per ciascuna classe.
    Se fused=True, model_or_client è un FusedSpaceFedClient (usa AE+classifier).
    Se fused=False, è un modello standard (FedAvg/FedProx/Scaffold).
    """
    correct = np.zeros(num_classes, dtype=np.int64)
    total   = np.zeros(num_classes, dtype=np.int64)

    device = next(model_or_client.parameters()).device \
             if not fused else next(model_or_client.clf.parameters()).device

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if fused:
                # FusedSpaceFedClient
                tx, _ = model_or_client._ae_forward(x)
                fused_x = model_or_client._fuse(x, tx)
                logits = model_or_client.clf(fused_x)
            else:
                # Normale (FedAvg / FedProx / Scaffold)
                logits = model_or_client(x)

            preds = logits.argmax(1)

            for cls in range(num_classes):
                mask = (y == cls)
                total[cls] += mask.sum().item()
                correct[cls] += (preds[mask] == cls).sum().item()

    # Evita divisioni per zero
    acc = np.zeros(num_classes)
    for cls in range(num_classes):
        acc[cls] = correct[cls] / total[cls] if total[cls] > 0 else 0.0
    return acc

def f1_on_loader(model_or_client, loader, num_classes=62, fused=False):
    """
    Calcola F1-macro su un loader.
    """
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if fused:
                tx, _ = model_or_client._ae_forward(x)
                fused_x = model_or_client._fuse(x, tx)
                logits = model_or_client.clf(fused_x)
            else:
                logits = model_or_client(x)

            preds = logits.argmax(1)

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return f1_score(y_true, y_pred, average="macro")

def _grad_vector_from_model(model, loader, max_batches=1, device=device):
    model.train()
    model.zero_grad()

    batches_done = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        batches_done += 1
        if batches_done >= max_batches:
            break

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    return torch.cat(grads) if grads else torch.zeros(0, device=device)

def _grad_vector_from_fused_client(client, loader, max_batches=1, device=device):
    client.ae.eval()
    client.clf.train()
    client.clf.zero_grad()

    batches_done = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.no_grad():
            tx, _ = client._ae_forward(x)
            fused_x = client._fuse(x, tx)

        logits = client.clf(fused_x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        batches_done += 1
        if batches_done >= max_batches:
            break

    grads = []
    for p in client.clf.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))

    return torch.cat(grads) if grads else torch.zeros(0, device=device)

def gradient_dissimilarity_gamma(method_name, clients, max_batches=1):
    grad_list = []

    for c in clients:
        if method_name.startswith("FusedSpaceFed"):
            g = _grad_vector_from_fused_client(c, c.loader, max_batches=max_batches)
        else:
            g = _grad_vector_from_model(c.model, c.loader, max_batches=max_batches)

        if g.numel() > 0:
            grad_list.append(g)

    if len(grad_list) == 0:
        return 0.0

    G = torch.stack(grad_list, dim=0)
    mean_g = G.mean(dim=0, keepdim=True)
    diff = G - mean_g

    return (diff.pow(2).sum(dim=1).mean().item())

# ============
# 5) Experiment (metriche globali & locali)
# ============
def run_experiment(
    strategy="dirichlet",
    alpha=0.05,
    classes_per_client=1,
    num_clients=3400,       # tutti i "client" logici
    rounds=100,             # 100 round
    local_epochs=3,         # 3 epoche per round
    batch_size=256,
    fusion_modes=("sum",),
    num_workers=0,
    lambda_sp=1e-3,
    warmup_encoder_epochs=2,
    balance_classes=False
):
    print(f"Strategy={strategy}, alpha={alpha}, classes_per_client={classes_per_client}")
    print(f"Rounds={rounds}, local_epochs={local_epochs}, batch_size={batch_size}, fusion_modes={fusion_modes}")

    # ================================
    # 1) LOAD DATA
    # ================================
    X_train, y_train, X_test, y_test = load_femnist_numpy()

    # ================================
    # OPTIONAL: BALANCING DELLE CLASSI
    # ================================
    if balance_classes:
        print("==> Balancing training classes (oversampling minority classes)...")

        unique, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()

        idx_balanced = []
        for cls in unique:
            cls_idx = np.where(y_train == cls)[0]
            reps = max_count // len(cls_idx) + 1
            new_idx = np.tile(cls_idx, reps)[:max_count]
            idx_balanced.append(new_idx)

        idx_balanced = np.concatenate(idx_balanced)
        np.random.shuffle(idx_balanced)

        X_train = X_train[idx_balanced]
        y_train = y_train[idx_balanced]

        print("Balanced class counts:", np.unique(y_train, return_counts=True))

    # ================================
    # 2) COSTRUISCO LE PARTIZIONI LOGICHE (client non ancora creati)
    # ================================
    train_part = ExtremeNonIIDRetina(
        X_train, y_train,
        train=True,
        strategy=strategy,
        alpha=alpha,
        num_clients=num_clients,
        classes_per_client=classes_per_client
    )

    test_global_ds = RetinaNPYDataset(X_test, y_test)
    global_test = DataLoader(
        test_global_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=False
    )

    test_part_local = ExtremeNonIIDRetina(
        X_test, y_test,
        train=False,
        strategy="dirichlet",
        alpha=0.5,
        num_clients=num_clients,
        classes_per_client=classes_per_client
    )

    # ================================
    # 3) METODI / SERVER / GLOBAL PARAMS
    # ================================
    # elenco metodi
    method_names = [
        "FedAvg",
        "FedProx",
        "Scaffold",
    ]
    for fm in fusion_modes:
        method_names.append(f"FusedSpaceFed_{fm}")
        method_names.append(f"FusedSpaceFed_{fm}_ENCWARM")
        method_names.append(f"FusedSpaceFed_{fm}_L2SP")

    # ==========================================
    # RUN SOLO UN METODO ALLA VOLTA
    # ==========================================
    method_names = ["FusedSpaceFed_sum_ENCWARM"]   # ← scegli il metodo da testare: "FedAvg", "FedProx", ecc.

    # server
    servers = {
        "FedAvg":   Server(num_clients),
        "FedProx":  Server(num_clients),
        "Scaffold": None,
    }

    # modello di riferimento per ScaffoldServer
    scaffold_proto = ResNet_CIFAR_V2(depth=20, num_classes=62, in_channels=1).to(device)
    servers["Scaffold"] = ScaffoldServer(num_clients, model_example=scaffold_proto)

    for fm in fusion_modes:
        servers[f"FusedSpaceFed_{fm}"]          = Server(num_clients)
        servers[f"FusedSpaceFed_{fm}_ENCWARM"]  = Server(num_clients)
        servers[f"FusedSpaceFed_{fm}_L2SP"]     = Server(num_clients)

    # global params per ogni metodo
    global_params = {m: None for m in method_names}
    global_clf_params = {}
    global_dec_params = {}
    for fm in fusion_modes:
        for suf in ["", "_ENCWARM", "_L2SP"]:
            key = f"FusedSpaceFed_{fm}{suf}"
            global_clf_params[key] = None
            global_dec_params[key] = None

    # ================================
    # 4) RESULTS STRUCTURE
    # ================================
    results = {
        m: {
            "train_acc_local": [],
            "test_acc_global": [],
            "test_acc_local": [],
            "test_f1_global": [],
            "test_f1_local": [],
            "losses": []
        } for m in method_names
    }

    best_global_acc = {m: 0.0 for m in method_names}
    best_per_class  = {m: None for m in method_names}
    best_global_f1  = {m: 0.0 for m in method_names}
    best_local_f1   = {m: 0.0 for m in method_names}
    best_params     = {m: None for m in method_names}

    # ================================
    # 5) FEDERATED TRAINING LOOP (client on-demand, 10% per round)
    # ================================
    pbar = tqdm(range(rounds), desc="FL Rounds")

    for r in pbar:
        round_post = {}

        total_clients = num_clients
        active_count = max(1, int(0.10 * total_clients))
        active_indices = np.random.choice(total_clients, active_count, replace=False).tolist()

        for method in method_names:

            # ========= CREA CLIENT ATTIVI SOLO PER QUESTO METODO / ROUND =========
            active_clients = []

            for cid in active_indices:
                loader = train_part.get_client_loader(
                    cid, batch_size=batch_size, num_workers=num_workers, shuffle=True
                )

                if method == "FedAvg":
                    c = FedAvgClient(cid, loader, num_classes=62)
                    if global_params[method] is not None:
                        c.set_parameters(global_params[method])

                elif method == "FedProx":
                    c = FedProxClient(cid, loader, num_classes=62, mu=0.01)
                    if global_params[method] is not None:
                        c.set_parameters(global_params[method])
                        c.set_global_params(global_params[method])

                elif method == "Scaffold":
                    c = ScaffoldClient(cid, loader, num_classes=62)
                    if global_params[method] is not None:
                        c.set_parameters(global_params[method])
                    c.set_server_control(servers["Scaffold"].c)

                else:
                    # metodi FusedSpaceFed
                    if method.startswith("FusedSpaceFed_") and method.endswith("_ENCWARM"):
                        fusion_mode = method.split("_")[1]  # "sum" o "concat"
                        warm = warmup_encoder_epochs
                        lam  = 0.0
                    elif method.startswith("FusedSpaceFed_") and method.endswith("_L2SP"):
                        fusion_mode = method.split("_")[1]
                        warm = 0
                        lam  = lambda_sp
                    else:
                        # FusedSpaceFed_sum "base"
                        fusion_mode = method.split("_")[1]
                        warm = 0
                        lam  = 0.0

                    c = FusedSpaceFedClient(
                        cid,
                        loader,
                        fusion_mode=fusion_mode,
                        num_classes=62,
                        warmup_encoder_epochs=warm,
                        lambda_sp=lam
                    )

                    if global_clf_params[method] is not None:
                        c.set_classifier_params(global_clf_params[method])
                    if global_dec_params[method] is not None:
                        c.set_decoder_params(global_dec_params[method])

                active_clients.append(c)

            # ========= PRE-TRAIN ACTIONS =========
            if method.startswith("FusedSpaceFed"):
                for c in active_clients:
                    c.on_round_start()

            if method == "FedProx" and global_params["FedProx"] is not None:
                for c in active_clients:
                    c.set_global_params(global_params["FedProx"])

            if method == "Scaffold":
                for c in active_clients:
                    c.set_server_control(servers["Scaffold"].c)

            # ========= LOCAL TRAINING =========
            if method in ("FedAvg", "FedProx", "Scaffold"):
                losses = [c.train(local_epochs) for c in active_clients]
                results[method]["losses"].append(float(np.mean(losses)) if losses else 0.0)
            else:
                losses = [c.train(local_epochs) for c in active_clients]
                if losses:
                    recon = float(np.mean([l[0] for l in losses]))
                    clf   = float(np.mean([l[1] for l in losses]))
                    results[method]["losses"].append((recon, clf))
                else:
                    results[method]["losses"].append((0.0, 0.0))

            # ========= AGGREGAZIONE + UPDATE GLOBAL PARAMS =========
            if method in ("FedAvg", "FedProx", "Scaffold"):
                param_dicts = [c.get_parameters() for c in active_clients]
                avg = servers[method]._avg_params(param_dicts)
                global_params[method] = avg
                for c in active_clients:
                    c.set_parameters(avg)

            else:
                clf_dicts = [c.get_classifier_params() for c in active_clients]
                dec_dicts = [c.get_decoder_params()   for c in active_clients]
                avg_clf = servers[method]._avg_params(clf_dicts)
                avg_dec = servers[method]._avg_params(dec_dicts)
                global_clf_params[method] = avg_clf
                global_dec_params[method] = avg_dec
                for c in active_clients:
                    c.set_classifier_params(avg_clf)
                    c.set_decoder_params(avg_dec)

            # ========= METRICHE =========
            with torch.no_grad():
                # train local acc (solo primi 3 client attivi)
                train_local = np.mean([
                    active_clients[i].evaluate(active_clients[i].loader)
                    for i in range(min(3, len(active_clients)))
                ]) if active_clients else 0.0

                # global acc (primi 5 client attivi sul test globale)
                test_global_acc = np.mean([
                    active_clients[i].evaluate(global_test)
                    for i in range(min(5, len(active_clients)))
                ]) if active_clients else 0.0

                # local non-iid test: stessi CID dei primi client attivi
                eval_count = min(5, len(active_clients))
                eval_cids = active_indices[:eval_count]
                local_loaders = [
                    test_part_local.get_client_loader(
                        cid, batch_size=256, num_workers=num_workers, shuffle=False
                    )
                    for cid in eval_cids
                ]
                test_local_acc = np.mean([
                    active_clients[i].evaluate(local_loaders[i])
                    for i in range(len(local_loaders))
                ]) if local_loaders else 0.0

                # F1 GLOBAL
                if method.startswith("FusedSpaceFed"):
                    f1_global = f1_on_loader(
                        active_clients[0],
                        global_test,
                        num_classes=62,
                        fused=True
                    ) if active_clients else 0.0
                else:
                    f1_global = f1_on_loader(
                        active_clients[0].model,
                        global_test,
                        num_classes=62,
                        fused=False
                    ) if active_clients else 0.0

                # F1 LOCAL
                if method.startswith("FusedSpaceFed"):
                    f1_local = np.mean([
                        f1_on_loader(active_clients[i], local_loaders[i], num_classes=62, fused=True)
                        for i in range(len(local_loaders))
                    ]) if local_loaders else 0.0
                else:
                    f1_local = np.mean([
                        f1_on_loader(active_clients[i].model, local_loaders[i], num_classes=62, fused=False)
                        for i in range(len(local_loaders))
                    ]) if local_loaders else 0.0

                # ========= SALVA METRICHE =========
                results[method]["train_acc_local"].append(float(train_local))
                results[method]["test_acc_global"].append(float(test_global_acc))
                results[method]["test_acc_local"].append(float(test_local_acc))
                results[method]["test_f1_global"].append(float(f1_global))
                results[method]["test_f1_local"].append(float(f1_local))

                # BEST GLOBAL ACC / PER-CLASS / PARAMS / F1
                if test_global_acc > best_global_acc[method]:
                    best_global_acc[method] = test_global_acc

                    if method.startswith("FusedSpaceFed"):
                        acc_pc = per_class_accuracy(
                            active_clients[0],
                            global_test,
                            num_classes=62,
                            fused=True
                        ) if active_clients else None
                    else:
                        acc_pc = per_class_accuracy(
                            active_clients[0].model,
                            global_test,
                            num_classes=62,
                            fused=False
                        ) if active_clients else None

                    best_per_class[method] = acc_pc

                    # salva best params globali
                    if method.startswith("FusedSpaceFed"):
                        best_params[method] = global_clf_params[method]
                    else:
                        best_params[method] = global_params[method]

                if f1_global > best_global_f1[method]:
                    best_global_f1[method] = f1_global

                round_post[f"{method}_G"]   = f"{test_global_acc:.3f}"
                round_post[f"{method}_L"]   = f"{test_local_acc:.3f}"
                round_post[f"{method}_F1G"] = f"{f1_global:.3f}"
                round_post[f"{method}_F1L"] = f"{f1_local:.3f}"

            # alla fine del metodo, liberiamo i client attivi
            del active_clients
            torch.cuda.empty_cache()

        pbar.set_postfix(round_post)

    # ================================
    # 6) GRUPPI PER GAMMA
    # ================================
    groups = {m: [] for m in method_names}
    gamma_clients_per_method = min(5, num_clients)

    for method in method_names:
        for cid in range(gamma_clients_per_method):
            loader = train_part.get_client_loader(
                cid, batch_size=batch_size, num_workers=num_workers, shuffle=True
            )

            if method == "FedAvg":
                c = FedAvgClient(cid, loader, num_classes=62)
                if best_params[method] is not None:
                    c.set_parameters(best_params[method])

            elif method == "FedProx":
                c = FedProxClient(cid, loader, num_classes=62, mu=0.01)
                if best_params[method] is not None:
                    c.set_parameters(best_params[method])
                    c.set_global_params(best_params[method])

            elif method == "Scaffold":
                c = ScaffoldClient(cid, loader, num_classes=62)
                if best_params[method] is not None:
                    c.set_parameters(best_params[method])
                c.set_server_control(servers["Scaffold"].c)

            else:
                if method.startswith("FusedSpaceFed_") and method.endswith("_ENCWARM"):
                    fusion_mode = method.split("_")[1]
                    warm = warmup_encoder_epochs
                    lam  = 0.0
                elif method.startswith("FusedSpaceFed_") and method.endswith("_L2SP"):
                    fusion_mode = method.split("_")[1]
                    warm = 0
                    lam  = lambda_sp
                else:
                    fusion_mode = method.split("_")[1]
                    warm = 0
                    lam  = 0.0

                c = FusedSpaceFedClient(
                    cid,
                    loader,
                    fusion_mode=fusion_mode,
                    num_classes=62,
                    warmup_encoder_epochs=warm,
                    lambda_sp=lam
                )

                if best_params[method] is not None:
                    c.set_classifier_params(best_params[method])

            groups[method].append(c)

    return results, best_global_acc, best_per_class, best_global_f1, best_local_f1, best_params, groups


# ============================================================================
# 6) MULTI-SEED + MULTI-ALPHA
# ============================================================================

import copy

balance_classes = False
alphas = np.arange(10, 10.06, 0.10)

SEEDS = list(range(1, 6))   # 5 SEED

# ==== STRUTTURA ACCUMULATORE PER ALPHA ====

final_stats = {
    alpha: {
        method: {
            "gamma": [],
            "best_acc_global": [],
            "best_acc_local": [],
            "best_f1_global": [],
            "balanced_acc": []
        }
        for method in [
            "FedAvg", "FedProx", "Scaffold",
            "FusedSpaceFed_sum",
            "FusedSpaceFed_sum_ENCWARM",
            "FusedSpaceFed_sum_L2SP"
        ]
    }
    for alpha in alphas
}

def balanced_accuracy(per_class_acc):
    return 0.0 if per_class_acc is None else float(np.mean(per_class_acc))


# ============================================================
# === LOOP ESTERNO SU α  → per ogni alpha farò TUTTI i seed ===
# ============================================================

for alpha_value in alphas:

    print("\n" + "="*100)
    print(f"########  PROCESSING α = {alpha_value:.2f}  ########")
    print("="*100)

    # Resetta l'accumulazione per questo α (anche se già inizializzata)
    # final_stats[alpha_value][method] mantiene una lista di 5 elementi (uno per seed)

    # ---------------------------------------------------------
    # === LOOP INTERNO SU SEED  → eseguo tutte le run =========
    # ---------------------------------------------------------

    for SEED in SEEDS:

        print(f"\n--- RUN: α = {alpha_value:.2f} | SEED = {SEED} ---")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        cfg = dict(
          strategy="dirichlet",
          alpha=float(alpha_value),
          classes_per_client=1,
          num_clients=50,      # o il valore che vuoi usare come totale client
          rounds=25,          # 100 round
          local_epochs=3,      # già 3 va bene
          batch_size=256,
          fusion_modes=("sum",),
          num_workers=0,
          lambda_sp=1e-3,
          warmup_encoder_epochs=2,
          balance_classes=balance_classes
          )

        (results,
         best_global_acc,
         best_per_class,
         best_global_f1,
         best_local_f1,
         best_params,
         groups) = run_experiment(**cfg)

        # ============================================================
        # === PLOT ACCURACY PER ROUND (Metodo singolo) ===============
        # ============================================================

        method = list(results.keys())[0]   # unico metodo eseguito

        rounds_axis = np.arange(len(results[method]["test_acc_global"]))

        plt.figure(figsize=(10,4))
        plt.plot(rounds_axis, results[method]["test_acc_global"], label="Global Acc")
        plt.plot(rounds_axis, results[method]["test_acc_local"],  label="Local Acc (non-IID)")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy per Round — {method}")
        plt.grid(True)
        plt.legend()
        plt.show()

        # --- Calcolo gamma
        gamma_results = {}
        for method, clients in groups.items():
            if best_params[method] is not None:
                if method.startswith("FusedSpaceFed"):
                    for c in clients:
                        c.set_classifier_params(best_params[method])
                else:
                    for c in clients:
                        c.set_parameters(best_params[method])

            gamma_val = gradient_dissimilarity_gamma(method, clients, max_batches=1)
            gamma_results[method] = float(gamma_val)

        # --- Salvataggio dei risultati in final_stats
        for method in results.keys():

            local_best = np.max(results[method]["test_acc_local"])
            ba = balanced_accuracy(best_per_class[method])

            final_stats[alpha_value][method]["gamma"].append(gamma_results[method])
            final_stats[alpha_value][method]["best_acc_global"].append(best_global_acc[method])
            final_stats[alpha_value][method]["best_acc_local"].append(local_best)
            final_stats[alpha_value][method]["best_f1_global"].append(best_global_f1[method])
            final_stats[alpha_value][method]["balanced_acc"].append(ba)

    # --------------------------------------------------------------
    # === DOPO TUTTI I SEED → STAMPA MEDIALE PER QUESTO α ===========
    # --------------------------------------------------------------

    print("\n\n################################################")
    print(f"###   RISULTATI MEDIATI SU SEED PER α = {alpha_value:.2f}   ###")
    print("################################################")

    for method in results.keys():

        stats = final_stats[alpha_value][method]

        mean_gamma = np.mean(stats["gamma"])
        mean_acc   = np.mean(stats["best_acc_global"])
        mean_local = np.mean(stats["best_acc_local"])
        mean_f1    = np.mean(stats["best_f1_global"])
        mean_bal   = np.mean(stats["balanced_acc"])

        print(f"\n{method}:")
        print(f"   Γ(x) medio:           {mean_gamma:.6f}")
        print(f"   Acc globale media:     {mean_acc:.4f}")
        print(f"   Acc locale media:      {mean_local:.4f}")
        print(f"   F1 globale media:      {mean_f1:.4f}")
        print(f"   Balanced Accuracy:     {mean_bal:.4f}")

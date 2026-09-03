"""Deterministic training and full-scene sliding-window prediction."""
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import config as cfg
from model import UNet, CombinedSegmentationLoss
from preprocessing import MultiCityTileDataset, make_model_input, window_starts


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def positive_weight(scenes, city_ids):
    positive = sum(int((scenes[c].truth * scenes[c].valid).sum()) for c in city_ids)
    valid = sum(int(scenes[c].valid.sum()) for c in city_ids)
    if positive <= 0: raise ValueError("Training cities contain no positive pixels")
    return float(np.clip(np.sqrt((valid - positive) / positive), 1, cfg.MAX_POSITIVE_WEIGHT))


def train_model(test_city, train_cities, stage, epochs, scenes, active_bands, checkpoint_tag):
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(str(stage))) % 100000
    seed = cfg.RANDOM_SEED + int(test_city) + offset; set_seed(seed)
    dataset = MultiCityTileDataset(scenes, train_cities, active_bands, True, seed)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                        pin_memory=cfg.DEVICE.type == "cuda", drop_last=False)
    model = UNet(len(active_bands) + 5).to(cfg.DEVICE)
    pweight = positive_weight(scenes, train_cities)
    criterion = CombinedSegmentationLoss(pweight, cfg.TVERSKY_ALPHA, cfg.TVERSKY_BETA,
                                         cfg.TVERSKY_WEIGHT, cfg.NEGATIVE_PIXEL_WEIGHT).to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3)
    amp = bool(cfg.USE_AMP and cfg.DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp); history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.; count = 0; start = time.time()
        for features, targets, valid in loader:
            features, targets, valid = features.to(cfg.DEVICE), targets.to(cfg.DEVICE), valid.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp): loss = criterion(model(features), targets, valid)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += float(loss.item()) * features.shape[0]; count += features.shape[0]
        epoch_loss = total / max(count, 1); scheduler.step(epoch_loss)
        history.append({"Fold_test_city": test_city, "Stage": stage, "Epoch": epoch, "Train_loss": epoch_loss,
                        "Learning_rate": optimizer.param_groups[0]["lr"], "Seconds": time.time() - start})
        print(f"{stage}: epoch {epoch:02d}/{epochs}, loss={epoch_loss:.6f}")
    path = cfg.MODEL_DIR / f"UNet_OOF_{checkpoint_tag}.pth"
    torch.save({"state_dict": model.state_dict(), "test_city": test_city, "training_cities": train_cities,
                "model_band_names": cfg.MODEL_BAND_NAMES, "active_band_names": active_bands, "epochs": epochs,
                "positive_weight": pweight, "loss": "negative-weighted BCE + 0.25*Tversky(alpha=0.8,beta=0.2)"}, path)
    return model, history, path


def predict_full_scene(model, scene, active_bands):
    model.eval(); accum = np.zeros((scene.height, scene.width), np.float32); overlap = np.zeros_like(accum, np.uint16)
    with torch.inference_mode():
        for y in window_starts(scene.height, cfg.TILE_SIZE, cfg.TEST_STRIDE):
            for x in window_starts(scene.width, cfg.TILE_SIZE, cfg.TEST_STRIDE):
                y2, x2 = min(y + cfg.TILE_SIZE, scene.height), min(x + cfg.TILE_SIZE, scene.width)
                spectral = scene.spectral[:, y:y2, x:x2]; h, w = spectral.shape[1:]
                if h < cfg.TILE_SIZE or w < cfg.TILE_SIZE:
                    spectral = np.pad(spectral, ((0, 0), (0, cfg.TILE_SIZE-h), (0, cfg.TILE_SIZE-w)), mode="edge")
                tensor = torch.from_numpy(make_model_input(spectral, scene, active_bands)).unsqueeze(0).to(cfg.DEVICE)
                probability = torch.sigmoid(model(tensor))[0, 0, :h, :w].cpu().numpy()
                accum[y:y2, x:x2] += probability; overlap[y:y2, x:x2] += 1
    if np.any(overlap == 0): raise RuntimeError(f"[{scene.city_id}] Inference left uncovered pixels")
    return accum / overlap.astype(np.float32)

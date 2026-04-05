import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
models = tf.keras.models

IMG_SIZE     = 256
MAX_NORMAL   = 3000
MAX_ABNORMAL = 100

NORMAL_DIR     = "data/normal/"
ABNORMAL_DIR   = "data/abnormal/"
MODEL_PATH     = "models/autoencoder.keras"
THRESHOLD_PATH = "models/threshold.npy"

PREPROCESS_CFG = {
    "use_adaptive_clahe" : True,
    "use_denoising"      : True,
    "denoise_ksize"      : (3, 3),
    "clahe_tile"         : (8, 8),
    "clahe_clip_default" : 3.0,
}



def apply_clahe(image_gray, cfg=PREPROCESS_CFG):
    if cfg["use_adaptive_clahe"]:
        variance   = float(np.var(image_gray.astype(np.float32)))
        clip_limit = float(np.clip(4.0 - (variance / 5000.0) * 2.5, 1.5, 4.0))
    else:
        clip_limit = cfg["clahe_clip_default"]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=cfg["clahe_tile"])
    return clahe.apply(image_gray)


def preprocess_image(img_gray, img_size=IMG_SIZE, cfg=PREPROCESS_CFG):
    if cfg["use_denoising"]:
        img_gray = cv2.GaussianBlur(img_gray, cfg["denoise_ksize"], 0)
    img_gray = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_gray = apply_clahe(img_gray, cfg)
    img = img_gray.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)


def load_images(folder_path, max_count=None):
    images = []
    files  = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if max_count:
        files = files[:max_count]
    print(f"  Loading {len(files)} from '{folder_path}' ...")
    for file in files:
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(preprocess_image(img))
    arr = np.array(images, dtype=np.float32)
    print(f"  Shape: {arr.shape}")
    return arr



def main():
    print("=" * 60)
    print("  THRESHOLD RECALIBRATION")
    print("=" * 60)

    print("\n[1/3] Loading model ...")
    model     = models.load_model(MODEL_PATH, compile=False)
    old_thresh = float(np.load(THRESHOLD_PATH))
    print(f"  Current threshold : {old_thresh:.6f}  (too low)")

    print("\n[2/3] Loading images ...")
    normal_images   = load_images(NORMAL_DIR,   max_count=MAX_NORMAL)
    abnormal_images = load_images(ABNORMAL_DIR, max_count=MAX_ABNORMAL)

    print("\n[3/3] Computing errors ...")
    recon_n = model.predict(normal_images,   verbose=1)
    recon_a = model.predict(abnormal_images, verbose=1)

    errors_n = np.mean((normal_images   - recon_n) ** 2, axis=(1, 2, 3))
    errors_a = np.mean((abnormal_images - recon_a) ** 2, axis=(1, 2, 3))

    print(f"\n  Normal   — min:{errors_n.min():.6f}  "
          f"mean:{errors_n.mean():.6f}  max:{errors_n.max():.6f}")
    print(f"  Abnormal — min:{errors_a.min():.6f}  "
          f"mean:{errors_a.mean():.6f}  max:{errors_a.max():.6f}")

    # ── Scan candidates ───────────────────────────────────────────────────────
    y_true     = np.array([0]*len(errors_n) + [1]*len(errors_a))
    all_errors = np.concatenate([errors_n, errors_a])
    candidates = np.linspace(all_errors.min(), all_errors.max(), 500)

    results = []
    for t in candidates:
        y_pred = (all_errors > t).astype(int)
        tp = np.sum((y_pred==1) & (y_true==1))
        fp = np.sum((y_pred==1) & (y_true==0))
        fn = np.sum((y_pred==0) & (y_true==1))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2*prec*rec / (prec + rec + 1e-8)
        results.append((t, prec, rec, f1))

    # ── Print table of interesting thresholds ─────────────────────────────────
    print(f"\n  {'Threshold':>12}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "─" * 65)

    # Show thresholds where recall >= 0.80 (still catching most weapons)
    shown = 0
    best_f1_entry     = max(results, key=lambda x: x[3])
    best_recall_entry = max(results, key=lambda x: x[2])

    for t, prec, rec, f1 in results:
        if rec >= 0.80:
            y_pred = (all_errors > t).astype(int)
            tp = int(np.sum((y_pred==1) & (y_true==1)))
            fp = int(np.sum((y_pred==1) & (y_true==0)))
            fn = int(np.sum((y_pred==0) & (y_true==1)))
            marker = ""
            if abs(t - best_f1_entry[0]) < 1e-8:
                marker = "  ← BEST F1"
            print(f"  {t:>12.6f}  {prec:>10.4f}  {rec:>8.4f}  "
                  f"{f1:>8.4f}  {tp:>5}  {fp:>5}  {fn:>5}{marker}")
            shown += 1
            if shown > 30:   # avoid flooding the console
                print("  ... (more rows above 0.80 recall)")
                break

    # ── Recommended threshold ─────────────────────────────────────────────────
    # Strategy: best F1 among candidates with Recall >= 0.85
    high_recall = [(t, p, r, f) for t, p, r, f in results if r >= 0.85]
    if high_recall:
        chosen = max(high_recall, key=lambda x: x[3])
    else:
        chosen = best_f1_entry

    new_thresh, new_prec, new_rec, new_f1 = chosen

    print(f"\n{'='*60}")
    print(f"  OLD threshold : {old_thresh:.6f}  "
          f"→  Prec=0.2257  Rec=1.0000")
    print(f"  NEW threshold : {new_thresh:.6f}  "
          f"→  Prec={new_prec:.4f}  Rec={new_rec:.4f}  F1={new_f1:.4f}")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    confirm = input(f"\n  Save new threshold {new_thresh:.6f}? (y/n): ").strip().lower()
    if confirm == 'y':
        np.save(THRESHOLD_PATH, new_thresh)
        print(f"  ✓ Threshold saved → {THRESHOLD_PATH}")
        print(f"  Now run: python Test.py --index 23")
    else:
        print("  Threshold NOT saved. Original kept.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ts  = [r[0] for r in results]
    plt.figure(figsize=(11, 4))
    plt.plot(ts, [r[1] for r in results], label='Precision', color='steelblue', lw=2)
    plt.plot(ts, [r[2] for r in results], label='Recall',    color='tomato',    lw=2)
    plt.plot(ts, [r[3] for r in results], label='F1',        color='green',     lw=2)
    plt.axvline(old_thresh, color='orange', ls='--', lw=1.5,
                label=f'Old: {old_thresh:.5f}')
    plt.axvline(new_thresh, color='black',  ls='--', lw=1.5,
                label=f'New: {new_thresh:.5f}')
    plt.title('Threshold Scan — Precision / Recall / F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/recalibration.png', dpi=150)
    print("  Plot saved → outputs/recalibration.png")
    plt.show()


if __name__ == "__main__":
    main()
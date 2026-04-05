import os
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
models = tf.keras.models
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix)
from scipy.ndimage import label as scipy_label
from utils.sam_utils import segment_with_sam


IMG_SIZE     = 256          
MAX_NORMAL   = 3000         
MAX_ABNORMAL = 100          

NORMAL_DIR     = "data/normal/"
ABNORMAL_DIR   = "data/abnormal/"
OUTPUT_DIR     = "outputs/"
MODEL_PATH     = "models/autoencoder.keras"
THRESHOLD_PATH = "models/threshold.npy"

# ✚ Preprocessing config (must match Train.py)
PREPROCESS_CFG = {
    "use_adaptive_clahe" : True,
    "use_denoising"      : True,
    "denoise_ksize"      : (3, 3),
    "clahe_tile"         : (8, 8),
    "clahe_clip_default" : 3.0,
}

# ✚ Cluster config
CLUSTER_CFG = {
    "heat_threshold"    : 0.30,   # lower than before — catches subtle anomalies
    "min_pixels"        : 15,     # minimum cluster size
    "top_n_clusters"    : 8,      # keep only top-N most significant
    "high_heat_cutoff"  : 0.60,   # above this → RED (high risk)
                                  # below this → BLUE (medium risk)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_clahe(image_gray, cfg=PREPROCESS_CFG):
    """✚ Adaptive clip limit based on image variance."""
    if cfg["use_adaptive_clahe"]:
        variance   = float(np.var(image_gray.astype(np.float32)))
        clip_limit = float(np.clip(4.0 - (variance / 5000.0) * 2.5, 1.5, 4.0))
    else:
        clip_limit = cfg["clahe_clip_default"]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=cfg["clahe_tile"])
    return clahe.apply(image_gray)


def preprocess_image(img_gray, img_size=IMG_SIZE, cfg=PREPROCESS_CFG):
    """Full pipeline: denoise → resize → adaptive CLAHE → normalize."""
    if cfg["use_denoising"]:
        img_gray = cv2.GaussianBlur(img_gray, cfg["denoise_ksize"], 0)
    img_gray = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_gray = apply_clahe(img_gray, cfg)
    img      = img_gray.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)


def load_images(folder_path, max_count=None, img_size=IMG_SIZE, cfg=PREPROCESS_CFG):
    """Original load_images — now uses enhanced preprocess_image."""
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
        images.append(preprocess_image(img, img_size, cfg))
    arr = np.array(images, dtype=np.float32)
    print(f"  Shape: {arr.shape}")
    return arr



def generate_heatmap(original, reconstructed):
    diff = np.abs(original - reconstructed).squeeze()

    # Step 1 — smooth noise
    diff = cv2.GaussianBlur(diff, (7, 7), 0)

    # Step 2 — percentile normalization (original used /max which is fragile)
    p99  = np.percentile(diff, 99)
    heatmap = np.clip(diff / (p99 + 1e-8), 0, 1).astype(np.float32)

    # Step 3 — suppress weak noise (below 20th percentile → zero)
    noise_floor = np.percentile(heatmap, 20)
    heatmap[heatmap < noise_floor] = 0.0

    # Step 4 — re-normalize after suppression
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap



def detect_density_clusters(heatmap, cfg=CLUSTER_CFG):
    binary_mask               = heatmap > cfg["heat_threshold"]
    labeled_mask, total_found = scipy_label(binary_mask)

    raw_clusters = []
    for cid in range(1, total_found + 1):
        coords = np.argwhere(labeled_mask == cid)
        if len(coords) < cfg["min_pixels"]:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        mean_heat    = float(heatmap[labeled_mask == cid].mean())
        raw_clusters.append({
            "id"          : cid,
            "pixel_count" : len(coords),
            "bbox"        : (x_min, y_min, x_max, y_max),
            "center"      : (int((x_min+x_max)/2), int((y_min+y_max)/2)),
            "mean_heat"   : mean_heat,
            # ✚ composite ranking score
            "score"       : len(coords) * mean_heat,
            # ✚ severity tag for color coding
            "severity"    : "HIGH" if mean_heat >= cfg["high_heat_cutoff"] else "MEDIUM",
        })

    # ✚ Rank by composite score, keep top N
    ranked     = sorted(raw_clusters, key=lambda c: c["score"], reverse=True)
    top        = ranked[:cfg["top_n_clusters"]]
    valid_ids  = {c["id"] for c in top}
    clean_mask = np.isin(labeled_mask, list(valid_ids)).astype(np.uint8)

    return len(top), top, (heatmap > cfg["heat_threshold"]), clean_mask



def draw_colored_overlay(original_img, cluster_info, sam_mask=None):
    vis = (original_img * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # ✚ SAM mask overlay first (so boxes draw on top)
    if sam_mask is not None:
        # Resize mask to match image if needed
        if sam_mask.shape[:2] != vis.shape[:2]:
            sam_mask_r = cv2.resize(
                sam_mask.astype(np.uint8),
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            sam_mask_r = sam_mask.astype(np.uint8)
        green_layer = np.zeros_like(vis)
        green_layer[:, :, 1] = 180   # green channel
        mask_bool = sam_mask_r > 0
        vis[mask_bool] = cv2.addWeighted(
            vis, 0.55, green_layer, 0.45, 0
        )[mask_bool]

    # ✚ Draw clusters with RED/BLUE color coding
    for c in cluster_info:
        x1, y1, x2, y2 = c["bbox"]
        cx, cy         = c["center"]

        if c["severity"] == "HIGH":
            box_color  = (0,  0,  220)   # RED  (BGR)
            fill_color = (0,  0,  255)
            dot_color  = (0,  255, 255)  # yellow dot
        else:
            box_color  = (200, 80,  0)   # BLUE (BGR)
            fill_color = (220, 120, 0)
            dot_color  = (255, 200, 0)

        # ✚ Semi-transparent filled rectangle
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        vis = cv2.addWeighted(vis, 0.80, overlay, 0.20, 0)

        # Solid border
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(vis, (cx, cy), 5, dot_color, -1)

        # Label: severity + heat score
        label = f"{c['severity'][0]}:{c['mean_heat']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(vis, (x1, max(y1-th-6, 0)),
                      (x1+tw+4, max(y1-2, th)), (30, 30, 30), -1)
        cv2.putText(vis, label, (x1+2, max(y1-4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)



def compute_risk_score(error, threshold, cluster_info=None):
    ratio = error / (threshold + 1e-8)

    # Component 1: error-based
    if ratio <= 1.0:
        error_score = ratio * 30.0
    else:
        error_score = 30.0 + min((ratio - 1.0) / 2.0, 1.0) * 30.0

    # Component 2: cluster count (0–20)
    if cluster_info:
        n_clusters    = len(cluster_info)
        cluster_score = min(n_clusters / 10.0, 1.0) * 20.0
    else:
        cluster_score = 0.0

    # Component 3: mean cluster intensity (0–20)
    if cluster_info:
        avg_heat      = np.mean([c["mean_heat"] for c in cluster_info])
        heat_score    = float(avg_heat) * 20.0
    else:
        heat_score = 0.0

    total = round(min(error_score + cluster_score + heat_score, 100.0), 2)
    return total, round(error_score, 2), round(cluster_score, 2), round(heat_score, 2)


def risk_label(score):
    if score < 40:
        return "LOW RISK",    "darkgreen"
    elif score < 65:
        return "MEDIUM RISK", "darkorange"
    else:
        return "HIGH RISK",   "darkred"


def compute_classification_metrics(errors_normal, errors_abnormal, threshold):
    
    y_true     = np.array([0]*len(errors_normal) + [1]*len(errors_abnormal))
    all_errors = np.concatenate([errors_normal, errors_abnormal])
    y_pred     = (all_errors > threshold).astype(int)
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred,    zero_division=0),
        f1_score(y_true, y_pred,        zero_division=0),
        confusion_matrix(y_true, y_pred),
        y_true, y_pred,
    )


def print_metrics(precision, recall, f1, cm):
   
    w = 52
    print("\n" + "─" * w)
    print("  CLASSIFICATION METRICS")
    print("─" * w)
    print(f"  {'Metric':<20} {'Value':>8}   {'Meaning'}")
    print("─" * w)
    print(f"  {'Precision':<20} {precision:>8.4f}   of flagged, how many real anomalies")
    print(f"  {'Recall':<20} {recall:>8.4f}   of all threats, how many caught")
    print(f"  {'F1 Score':<20} {f1:>8.4f}   harmonic mean (balance)")
    print("─" * w)
    print(f"  CONFUSION MATRIX")
    print(f"  {'':>20}  Pred Normal   Pred Anomaly")
    print(f"  {'True Normal':<20}  {cm[0,0]:>11}   {cm[0,1]:>12}")
    print(f"  {'True Anomaly':<20}  {cm[1,0]:>11}   {cm[1,1]:>12}")
    print("─" * w)


def visualize_result(original, reconstructed, error, threshold,
                     precision, recall, f1, idx,
                     cluster_info, sam_image, sam_mask,
                     save_name="result.png"):
    
    risk, e_score, c_score, h_score = compute_risk_score(
        error, threshold, cluster_info
    )
    rlabel, rcolor = risk_label(risk)
    heatmap        = generate_heatmap(original, reconstructed)
    original_sq    = original.squeeze()
    recon_sq       = reconstructed.squeeze()

    overlay = draw_colored_overlay(original_sq, cluster_info, sam_mask)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 6), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(1, 5, figure=fig,
                            left=0.02, right=0.98,
                            top=0.82, bottom=0.05,
                            wspace=0.06)

    panel_style = dict(facecolor='#1a1a1a')

    # ── Panel 1: Original ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0], **panel_style)
    ax1.imshow(original_sq, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Original X-ray", color='white', fontsize=11, pad=6)
    ax1.axis('off')
    _frame(ax1, '#444')

    # ── Panel 2: Reconstructed ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], **panel_style)
    ax2.imshow(recon_sq, cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Reconstructed", color='#aaa', fontsize=11, pad=6)
    ax2.axis('off')
    _frame(ax2, '#444')

    # ── Panel 3: Heatmap ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], **panel_style)
    im3 = ax3.imshow(heatmap, cmap='inferno', vmin=0, vmax=1)
    ax3.set_title("Anomaly Heatmap", color='#ff7043', fontsize=11, pad=6)
    ax3.axis('off')
    _frame(ax3, '#ff7043')
    cb = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors='white', labelsize=7)

    # ── Panel 4: Colored overlay ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], **panel_style)
    ax4.imshow(overlay)
    n_high = sum(1 for c in cluster_info if c["severity"] == "HIGH")
    n_med  = len(cluster_info) - n_high
    ax4.set_title(
        f"Anomaly Regions  🔴{n_high} HIGH  🔵{n_med} MED",
        color='white', fontsize=10, pad=6
    )
    ax4.axis('off')
    _frame(ax4, rcolor)

    # ── Panel 5: Risk gauge + info card ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[4], facecolor='#111')
    ax5.axis('off')
    _draw_risk_gauge(ax5, risk, rlabel, rcolor,
                     error, threshold, e_score, c_score, h_score,
                     precision, recall, f1, len(cluster_info))

    # ── Super title ───────────────────────────────────────────────────────────
    pred_str = "ANOMALY DETECTED" if error > threshold else "NORMAL"
    fig.text(
        0.5, 0.93,
        f"Image [{idx}]  ·  {pred_str}  ·  Risk: {risk}/100  ·  {rlabel}  "
        f"·  Error: {error:.5f}  ·  Threshold: {threshold:.5f}  "
        f"·  Clusters: {len(cluster_info)}",
        ha='center', va='center',
        fontsize=10, fontweight='bold', color=rcolor
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.text(0.25, 0.97,
             "🔴 RED = High Anomaly Region   🔵 BLUE = Medium Anomaly Region"
             "   🟢 SAM Mask Overlay",
             ha='center', fontsize=8, color='#ccc')

    out = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  ✓ Saved → {out}")
    plt.show()
    plt.close()


def _frame(ax, color, lw=1.5):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


def _draw_risk_gauge(ax, risk, rlabel, rcolor,
                     error, threshold, e_score, c_score, h_score,
                     precision, recall, f1, n_clusters):
    # Gauge bar (horizontal)
    bar_w  = 0.80
    bar_h  = 0.09
    bar_x  = 0.10
    bar_y  = 0.74

    # Background track
    bg = mpatches.FancyBboxPatch(
        (bar_x, bar_y), bar_w, bar_h,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor='#333', transform=ax.transAxes
    )
    ax.add_patch(bg)

    # Filled portion
    fill_w = bar_w * (risk / 100.0)
    fill_color = rcolor
    fg = mpatches.FancyBboxPatch(
        (bar_x, bar_y), fill_w, bar_h,
        boxstyle="round,pad=0.01",
        linewidth=0, facecolor=fill_color, transform=ax.transAxes,
        alpha=0.85
    )
    ax.add_patch(fg)

    # Threshold marker at 50%
    mid_x = bar_x + bar_w * 0.5
    ax.plot([mid_x, mid_x],
            [bar_y, bar_y + bar_h + 0.04],
            color='white', lw=1.0, ls='--', alpha=0.5,
            transform=ax.transAxes)
    
        # Risk score text
    ax.text(0.5, bar_y + bar_h + 0.04, f"RISK SCORE: {risk}/100",
            ha='center', va='bottom', transform=ax.transAxes,
            color='white', fontsize=13, fontweight='bold')
    ax.text(0.5, bar_y - 0.06, rlabel,
            ha='center', va='top', transform=ax.transAxes,
            color=rcolor, fontsize=12, fontweight='bold')

    # Breakdown
    lines = [
        ("Error component",   f"{e_score:.1f}/60"),
        ("Cluster component", f"{c_score:.1f}/20"),
        ("Intensity component", f"{h_score:.1f}/20"),
        ("", ""),
        ("Reconstruction Error", f"{error:.6f}"),
        ("Threshold",           f"{threshold:.6f}"),
        ("Clusters found",      f"{n_clusters}"),
        ("", ""),
        ("Precision",   f"{precision:.4f}"),
        ("Recall",      f"{recall:.4f}"),
        ("F1 Score",    f"{f1:.4f}"),
    ]
    y_pos = 0.64
    for key, val in lines:
        if key == "":
            y_pos -= 0.025
            continue
        ax.text(0.08, y_pos, key + ":", ha='left', va='top',
                transform=ax.transAxes, color='#aaa', fontsize=8)
        ax.text(0.92, y_pos, val, ha='right', va='top',
                transform=ax.transAxes, color='white', fontsize=8,
                fontweight='bold')
        y_pos -= 0.075

    # Scale labels
    ax.text(bar_x,          bar_y - 0.015, "0",   ha='left',   va='top',
            transform=ax.transAxes, color='#888', fontsize=7)
    ax.text(bar_x + bar_w/2, bar_y - 0.015, "50", ha='center', va='top',
            transform=ax.transAxes, color='#888', fontsize=7)
    ax.text(bar_x + bar_w,  bar_y - 0.015, "100", ha='right',  va='top',
            transform=ax.transAxes, color='#888', fontsize=7)


def analyze_image(idx, images, reconstructed, errors,
                  threshold, precision, recall, f1,
                  prefix="result"):
    total = len(images)
    if idx < 0 or idx >= total:
        print(f"  [ERROR] Index {idx} out of range. Valid: 0 to {total-1}")
        return

    original = images[idx]
    recon    = reconstructed[idx]
    error    = float(errors[idx])
    label    = "ANOMALY" if error > threshold else "NORMAL"

    # Heatmap + clusters
    heatmap                          = generate_heatmap(original, recon)
    num_c, cluster_info, _, _        = detect_density_clusters(heatmap)

    # ✚ Enhanced risk score
    risk, e_sc, c_sc, h_sc = compute_risk_score(error, threshold, cluster_info)
    rlabel, rcolor         = risk_label(risk)

    # SAM segmentation (original integration preserved)
    sam_image, sam_mask = segment_with_sam(original.squeeze(), heatmap)

    # Console output
    print(f"\n{'─'*60}")
    print(f"  Image [{idx}/{total-1}]  →  {label}")
    print(f"  Error       : {error:.6f}  |  Threshold : {threshold:.6f}")
    print(f"  Risk Score  : {risk}/100  →  {rlabel}")
    print(f"  Breakdown   : error={e_sc}  clusters={c_sc}  intensity={h_sc}")
    print(f"  Clusters    : {num_c}")
    for c in cluster_info:
        sev = "🔴" if c["severity"] == "HIGH" else "🔵"
        print(f"    {sev} [{c['id']:02d}] pixels={c['pixel_count']:>4}  "
              f"center={c['center']}  heat={c['mean_heat']:.3f}  "
              f"score={c['score']:.1f}")

    visualize_result(
        original, recon, error, threshold,
        precision, recall, f1, idx,
        cluster_info, sam_image, sam_mask,
        save_name=f"{prefix}_idx{idx}.png"
    )


def main():
    parser = argparse.ArgumentParser(description="Test anomaly detection model.")
    parser.add_argument("--index",    type=int, default=0,
                        help="Abnormal image index (default: 0)")
    parser.add_argument("--all",      action="store_true",
                        help="Analyze all abnormal images")
    # ✚ Cross-dataset: test on any folder without changing normal/abnormal dirs
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional: test on a different dataset folder "
                             "(e.g. data/test or data/pidray)")
    parser.add_argument("--max_test", type=int, default=200,
                        help="Max images to load from --test_dir (default: 200)")
    args = parser.parse_args()

    print("=" * 65)
    print("  TEST — Enhanced Anomaly Detection  (Hackathon v4)")
    print("=" * 65)

    # Validate model + threshold exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at '{MODEL_PATH}'. Run Train.py first!")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"No threshold at '{THRESHOLD_PATH}'. Run Train.py first!")

    # [1] Load model + threshold
    print(f"\n[1/4] Loading model ...")
    model     = models.load_model(MODEL_PATH, compile=False)
    threshold = float(np.load(THRESHOLD_PATH))
    print(f"  ✓ Threshold (calibrated) : {threshold:.6f}")

    # [2] Load images
    print("\n[2/4] Loading images ...")
    normal_images   = load_images(NORMAL_DIR,   max_count=MAX_NORMAL)
    abnormal_images = load_images(ABNORMAL_DIR, max_count=MAX_ABNORMAL)
    print(f"  Abnormal range: 0 to {len(abnormal_images)-1}")

    # ✚ Cross-dataset: load from --test_dir if provided
    if args.test_dir:
        print(f"\n  ✚ Cross-dataset mode: loading from '{args.test_dir}' ...")
        test_images   = load_images(args.test_dir, max_count=args.max_test)
        test_label    = os.path.basename(args.test_dir.rstrip("/"))
    else:
        test_images = None
        test_label  = None

    # [3] Predict
    print("\n[3/4] Predicting ...")
    recon_normal   = model.predict(normal_images,   verbose=1)
    recon_abnormal = model.predict(abnormal_images, verbose=1)

    errors_normal   = np.mean((normal_images   - recon_normal)   ** 2, axis=(1,2,3))
    errors_abnormal = np.mean((abnormal_images - recon_abnormal) ** 2, axis=(1,2,3))

    print(f"\n  Normal   — mean:{errors_normal.mean():.6f}  "
          f"std:{errors_normal.std():.6f}")
    print(f"  Abnormal — mean:{errors_abnormal.mean():.6f}  "
          f"std:{errors_abnormal.std():.6f}")

    # [4] Metrics
    print("\n[4/4] Metrics ...")
    precision, recall, f1, cm, _, _ = compute_classification_metrics(
        errors_normal, errors_abnormal, threshold
    )
    print_metrics(precision, recall, f1, cm)

    # ✚ If cross-dataset, also run on that folder (no ground-truth, just inference)
    if test_images is not None and len(test_images) > 0:
        print(f"\n  ✚ Cross-dataset inference on '{test_label}' ...")
        recon_test   = model.predict(test_images, verbose=1)
        errors_test  = np.mean((test_images - recon_test) ** 2, axis=(1,2,3))
        n_anomalies  = int(np.sum(errors_test > threshold))
        print(f"  Dataset : {test_label} ({len(test_images)} images)")
        print(f"  Flagged as anomaly : {n_anomalies}/{len(test_images)}")

        # Visualize first anomaly found in test set
        anom_indices = np.where(errors_test > threshold)[0]
        if len(anom_indices) > 0:
            first_anom = int(anom_indices[0])
            print(f"  Visualizing first anomaly in test set (index {first_anom}) ...")
            analyze_image(
                first_anom, test_images, recon_test, errors_test,
                threshold, precision, recall, f1,
                prefix=f"cross_{test_label}"
            )

    # Analyze abnormal images
    if args.all:
        print(f"\n  Analyzing all {len(abnormal_images)} abnormal images ...")
        for i in range(len(abnormal_images)):
            analyze_image(i, abnormal_images, recon_abnormal, errors_abnormal,
                          threshold, precision, recall, f1)
    else:
        analyze_image(args.index, abnormal_images, recon_abnormal, errors_abnormal,
                      threshold, precision, recall, f1)

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
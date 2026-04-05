import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
layers  = tf.keras.layers
models  = tf.keras.models
from sklearn.model_selection import train_test_split


IMG_SIZE     = 256         
CHANNELS     = 1
EPOCHS       = 30           
BATCH_SIZE   = 16           


MAX_NORMAL   = 3000         
MAX_ABNORMAL = 100          

NORMAL_DIR   = "data/normal/"
ABNORMAL_DIR = "data/abnormal/"
MODEL_PATH     = "models/autoencoder.keras"
THRESHOLD_PATH = "models/threshold.npy"


PREPROCESS_CFG = {
    "use_adaptive_clahe" : True,   
    "use_denoising"      : True,    
    "denoise_ksize"      : (3, 3),  
    "clahe_tile"         : (8, 8),  
    "clahe_clip_default" : 3.0,    
}

for d in ["models/", "outputs/"]:
    os.makedirs(d, exist_ok=True)


def apply_clahe(image_gray, cfg=PREPROCESS_CFG):
    if cfg["use_adaptive_clahe"]:
        variance   = float(np.var(image_gray.astype(np.float32)))
        clip_limit = float(np.clip(4.0 - (variance / 5000.0) * 2.5, 1.5, 4.0))
    else:
        clip_limit = cfg["clahe_clip_default"]

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=cfg["clahe_tile"])
    return clahe.apply(image_gray)


def preprocess_image(img_gray, img_size=IMG_SIZE, cfg=PREPROCESS_CFG):
    if cfg["use_denoising"]:
        img_gray = cv2.GaussianBlur(img_gray, cfg["denoise_ksize"], 0)

    # Step 1 — resize (original)
    img_gray = cv2.resize(img_gray, (img_size, img_size),
                          interpolation=cv2.INTER_AREA)

    # Step 2 — CLAHE (enhanced)
    img_gray = apply_clahe(img_gray, cfg)

    # Step 3 — normalize + channel dim (original)
    img = img_gray.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)


def load_images(folder_path, img_size=IMG_SIZE, max_count=None, cfg=PREPROCESS_CFG):
    images = []
    files  = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if max_count:
        files = files[:max_count]

    print(f"  Found {len(files)} images in '{folder_path}' ...")
    for file in files:
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = preprocess_image(img, img_size, cfg)
        images.append(img)

    arr = np.array(images, dtype=np.float32)
    print(f"  Loaded: {arr.shape}")
    return arr


def augment_images(images):
    flipped_h = images[:, :, ::-1, :]
    flipped_v = images[:, ::-1, :, :]
    augmented = np.concatenate([images, flipped_h, flipped_v], axis=0)
    np.random.shuffle(augmented)
    print(f"  Augmented: {images.shape[0]} → {augmented.shape[0]} images")
    return augmented


def ssim_loss(y_true, y_pred):
    """Structural similarity loss — penalizes shape/texture differences."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def combined_loss(y_true, y_pred):
    """50% MSE + 50% SSIM — pixel accuracy + structural accuracy."""
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return 0.5 * mse + 0.5 * ssim

def conv_block(x, filters):
    """✚ Conv → BatchNorm → LeakyReLU (original used plain Conv → ReLU)."""
    x = layers.Conv2D(filters, (3, 3), padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    return x


def build_autoencoder(img_size=IMG_SIZE):
    inp = layers.Input(shape=(img_size, img_size, CHANNELS))

    # ── ENCODER (original 3 blocks + 1 new) ──────────────────────────────────
    x = conv_block(inp, 32);    x = conv_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)           # → 128

    x = conv_block(x, 64);     x = conv_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)           # → 64

    x = conv_block(x, 128);    x = conv_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)           # → 32

    x = conv_block(x, 256);    x = conv_block(x, 256)
    encoded = layers.MaxPooling2D((2, 2))(x)     # → 16  ← bottleneck

    # ── BOTTLENECK ────────────────────────────────────────────────────────────
    x = conv_block(encoded, 512)
    x = layers.Dropout(0.3)(x)                 
    x = conv_block(x, 256)

    # ── DECODER (mirrors encoder) ─────────────────────────────────────────────
    x = layers.UpSampling2D((2, 2))(x);  x = conv_block(x, 256); x = conv_block(x, 256)
    x = layers.UpSampling2D((2, 2))(x);  x = conv_block(x, 128); x = conv_block(x, 128)
    x = layers.UpSampling2D((2, 2))(x);  x = conv_block(x, 64);  x = conv_block(x, 64)
    x = layers.UpSampling2D((2, 2))(x);  x = conv_block(x, 32);  x = conv_block(x, 32)

    # ── OUTPUT (original sigmoid — unchanged) ─────────────────────────────────
    out = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inp, out, name='AutoencoderV4')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=combined_loss          # ✚ replaces plain 'mse'
    )
    return model


def calibrate_threshold(model, normal_val, abnormal_images):
    """
    Find the decision boundary that best separates normal from anomaly.
    Uses both sets so the threshold is grounded in actual error distributions.
    Falls back to best-F1 if no candidate achieves Precision >= 0.70.
    """
    print("\n  Calibrating threshold ...")

    recon_n  = model.predict(normal_val,      verbose=0)
    recon_a  = model.predict(abnormal_images, verbose=0)

    errors_n = np.mean((normal_val      - recon_n) ** 2, axis=(1, 2, 3))
    errors_a = np.mean((abnormal_images - recon_a) ** 2, axis=(1, 2, 3))

    print(f"  Normal   — min:{errors_n.min():.5f} "
          f"mean:{errors_n.mean():.5f} max:{errors_n.max():.5f}")
    print(f"  Abnormal — min:{errors_a.min():.5f} "
          f"mean:{errors_a.mean():.5f} max:{errors_a.max():.5f}")

    y_true     = np.array([0]*len(errors_n) + [1]*len(errors_a))
    all_errors = np.concatenate([errors_n, errors_a])
    candidates = np.linspace(all_errors.min(), all_errors.max(), 200)

    best_thresh, best_recall, best_f1 = candidates[0], 0.0, 0.0
    results = []

    for t in candidates:
        y_pred = (all_errors > t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        results.append((t, prec, rec, f1))
        if prec >= 0.70 and rec > best_recall:
            best_recall, best_thresh, best_f1 = rec, t, f1

    # Fallback to best F1 if precision floor never met
    if best_recall == 0.0:
        print("  [WARN] Precision floor not met — using best F1.")
        best_thresh, _, best_recall, best_f1 = max(results, key=lambda x: x[3])

    print(f"  Chosen threshold : {best_thresh:.6f}  "
          f"(Recall={best_recall:.3f}  F1={best_f1:.3f})")

    # Save calibration plot
    ts  = [r[0] for r in results]
    plt.figure(figsize=(10, 4))
    plt.plot(ts, [r[1] for r in results], label='Precision', color='steelblue', lw=2)
    plt.plot(ts, [r[2] for r in results], label='Recall',    color='tomato',    lw=2)
    plt.plot(ts, [r[3] for r in results], label='F1',        color='green',     lw=2)
    plt.axvline(best_thresh, color='black', linestyle='--',
                label=f'Chosen: {best_thresh:.5f}')
    plt.title('Threshold Calibration Scan')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/threshold_calibration.png', dpi=150)
    print("  Calibration plot → outputs/threshold_calibration.png")
    plt.close()

    return best_thresh


def main():
    # ✚ CLI: support cross-dataset via --data_dir
    parser = argparse.ArgumentParser(description="Train anomaly detection autoencoder.")
    parser.add_argument("--normal_dir",   default=NORMAL_DIR,
                        help=f"Normal images folder (default: {NORMAL_DIR})")
    parser.add_argument("--abnormal_dir", default=ABNORMAL_DIR,
                        help=f"Abnormal images folder (default: {ABNORMAL_DIR})")
    parser.add_argument("--max_normal",   type=int, default=MAX_NORMAL,
                        help=f"Max normal images to use (default: {MAX_NORMAL})")
    parser.add_argument("--max_abnormal", type=int, default=MAX_ABNORMAL,
                        help=f"Max abnormal images to use (default: {MAX_ABNORMAL})")
    args = parser.parse_args()

    print("=" * 65)
    print("  TRAIN — Enhanced Convolutional Autoencoder  (Hackathon v4)")
    print("=" * 65)
    print(f"  Normal dir   : {args.normal_dir}")
    print(f"  Abnormal dir : {args.abnormal_dir}")
    print(f"  Max normal   : {args.max_normal}")
    print(f"  Max abnormal : {args.max_abnormal}")

    # [1] Load images
    print("\n[1/5] Loading images ...")
    normal_images   = load_images(args.normal_dir,   max_count=args.max_normal)
    abnormal_images = load_images(args.abnormal_dir, max_count=args.max_abnormal)

    if len(normal_images) < 50:
        raise ValueError(f"Only {len(normal_images)} normal images found. Need >= 50.")
    if len(abnormal_images) < 5:
        raise ValueError(f"Only {len(abnormal_images)} abnormal images. Need >= 5 for calibration.")

    print(f"  Normal   : {normal_images.shape}")
    print(f"  Abnormal : {abnormal_images.shape}")

    # [2] Split normal into train / val
    print("\n[2/5] Splitting ...")
    X_train, X_val = train_test_split(normal_images, test_size=0.1, random_state=42)

    # [3] Augment training set
    print("\n[3/5] Augmenting ...")
    X_train = augment_images(X_train)
    print(f"  Train : {X_train.shape[0]} | Val : {X_val.shape[0]}")

    # [4] Build + train
    print("\n[4/5] Training ...")
    model = build_autoencoder()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_loss',
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n  ✓ Model saved → {MODEL_PATH}")

    # [5] Calibrate threshold
    print("\n[5/5] Calibrating threshold ...")
    threshold = calibrate_threshold(model, X_val, abnormal_images)
    np.save(THRESHOLD_PATH, threshold)
    print(f"  ✓ Threshold saved → {THRESHOLD_PATH}  (value: {threshold:.6f})")

    # Plot training loss
    plt.figure(figsize=(9, 4))
    plt.plot(history.history['loss'],     label='Train Loss', color='steelblue', lw=2)
    plt.plot(history.history['val_loss'], label='Val Loss',   color='tomato',    lw=2, linestyle='--')
    plt.title('Training Loss (MSE + SSIM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/training_loss.png', dpi=150)
    print("  ✓ Loss curve → outputs/training_loss.png")
    plt.show()

    print("\n✅ Training complete!  Now run:  python Test.py")


if __name__ == "__main__":
    main()
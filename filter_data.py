import os
import shutil


NORMAL_FOLDER = "data/normal/"
ABNORMAL_FOLDER = "data/abnormal/"

os.makedirs(NORMAL_FOLDER, exist_ok=True)
os.makedirs(ABNORMAL_FOLDER, exist_ok=True)


def filter_dataset(image_folder, label_folder, dataset_name=""):
    normal_count = 0
    abnormal_count = 0

    print(f"\nFiltering: {dataset_name}")

    for file in os.listdir(image_folder):
        if file.endswith((".jpg", ".png")):
            label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")

            label_path = os.path.join(label_folder, label_file)
            image_path = os.path.join(image_folder, file)

            if os.path.exists(label_path):
                if os.path.getsize(label_path) == 0:
                    shutil.copy(image_path, NORMAL_FOLDER)
                    normal_count += 1
                else:
                    shutil.copy(image_path, ABNORMAL_FOLDER)
                    abnormal_count += 1
            else:
                # No label → assume normal
                shutil.copy(image_path, NORMAL_FOLDER)
                normal_count += 1

    print(f"  Normal: {normal_count}")
    print(f"  Abnormal: {abnormal_count}")


# SIXRAY
filter_dataset(
    "sixray_v3/train/images/",
    "sixray_v3/train/labels/",
    "SIXRAY TRAIN"
)

filter_dataset(
    "sixray_v3/test/images/",
    "sixray_v3/test/labels/",
    "SIXRAY TEST"
)

# CARGOXRAY (ADD YOUR PATHS HERE)
filter_dataset(
    "train/images/",
    "train/labels/",
    "CARGO TRAIN"
)

filter_dataset(
    "test/images/",
    "test/labels/",
    "CARGO TEST"
)

print("\n✅ ALL DATA FILTERED SUCCESSFULLY")
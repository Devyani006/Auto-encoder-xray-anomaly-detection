from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2

SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)


def get_bounding_box_from_heatmap(heatmap, threshold=0.5):
    heatmap = (heatmap > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return np.array([x, y, x+w, y+h])


def segment_with_sam(image, heatmap):
    """
    image: original image (128x128 grayscale or RGB)
    heatmap: anomaly heatmap
    """

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = (image * 255).astype(np.uint8)

    predictor.set_image(image)

    box = get_bounding_box_from_heatmap(heatmap)

    if box is None:
        return image, None

    masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)

    return image, masks[0]
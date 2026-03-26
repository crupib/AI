# run_detr_resnet_mac_fixed.py

import os
# Disable Hugging Face auto safetensors conversion (thread patch)
os.environ["TRANSFORMERS_NO_AUTO_CONVERT"] = "1"

# -----------------------------
# Monkey-patch Transformers auto_conversion
# -----------------------------
import types
from transformers import safetensors_conversion

def no_auto_conversion(*args, **kwargs):
    return None

safetensors_conversion.auto_conversion = types.FunctionType(
    no_auto_conversion.__code__,
    globals(),
    name="auto_conversion"
)

# -----------------------------
# Standard imports
# -----------------------------
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests

# -----------------------------
# Load an example image
# -----------------------------
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# -----------------------------
# Load processor and model
# -----------------------------
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    revision="no_timm"
)

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    revision="no_timm",
    use_safetensors=False
)

# -----------------------------
# Run inference
# -----------------------------
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.9  # only keep confident detections
)[0]

# -----------------------------
# Draw results on the image
# -----------------------------
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    class_name = model.config.id2label[label.item()]
    print(f"Detected {class_name} with confidence {round(score.item(),3)} at location {box}")

    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)
    # Draw label
    draw.text((box[0], box[1] - 10), f"{class_name} {round(score.item(),3)}", fill="red")

# Show or save image
image.show()  # Opens Preview on macOS
# image.save("detr_output.jpg")  # Optional: save result

import torch
from transformers import AutoModelForImageClassification

# Load model
model_name = "model"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

# Create dummy input in shape [batch, channels, height, width]
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "food_classifier.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=11
)

print("✅ Model exported to food_classifier.onnx")

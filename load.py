from transformers import AutoModelForImageClassification, ViTImageProcessor

MODEL_PATH="./model"

model_name = "Falconsai/nsfw_image_detection"

model = AutoModelForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# 保存到指定目录
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)
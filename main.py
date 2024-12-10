# Use a pipeline as a high-level helper
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch
import flask
import base64
from io import BytesIO
import time

MODEL_PATH = "./model"

app = flask.Flask(__name__)


@app.route("/")
def index():
    # 首页，上传图片的界面
    return """
    <!DOCTYPE html>
    <html>
    <body>
    <form action="/predict" method="post" enctype="multipart/form-data">
        Select image to upload:
        <input type="file" name="image" id="image">
        <input type="submit" value="Upload Image" name="submit">
    </form>
    </body>
    </html>
    """


response_tpl = """
<!DOCTYPE html>
<html>
<body>

<img src="data:image/png;base64,{}" alt="image" width="500" height="600">
<form action="/predict" method="post" enctype="multipart/form-data">
    Select image to upload:
    <input type="file" name="image" id="image">
    <input type="submit" value="Upload Image" name="submit">
</form>

<p>Prediction: {}</p>
<p>Score: {}</p>
<p>Take {} seconds</p>

</body>
</html>
"""


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    img_file = flask.request.files["image"]

    # check if the post request has the file part
    if "image" not in flask.request.files:
        return "No file part"

    img = Image.open(img_file)
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label_idx = probabilities.argmax(-1).item()
    score = float(probabilities[0, predicted_label_idx].item())

    label = model.config.id2label[predicted_label_idx]

    # 转换图片为 Base64 编码
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    end = time.time()
    print("Take {} seconds".format(end-start))

    # 返回带图片和预测结果的 HTML
    return response_tpl.format(img_base64, label, score, end-start)


# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

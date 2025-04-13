from flask import Flask, request, render_template
from PIL import Image
from torchvision import transforms
import random

app = Flask(__name__)

# Dummy labels
binary_labels = ["Fake", "Real"]
category_labels = ["Animal", "Human", "Vehicle"]

# Dummy transform (simulate preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    img = Image.open(request.files['image'].stream).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Simulated model input

    # 🔮 Simulated prediction (replace later with model)
    binary_pred = random.choice(binary_labels)
    class_pred = random.choice(category_labels)

    return render_template('results.html',
                           real_or_fake=binary_pred,
                           category=class_pred)

if __name__ == "__main__":
    app.run(debug=True)

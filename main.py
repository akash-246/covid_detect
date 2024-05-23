import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from torchvision import transforms

app = FastAPI()

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class labels
custom_labels = {0: 'covid', 1: 'viral', 2: 'normal'}  # Define your custom class labels

# Load your custom model
model = torch.load("/Users/akashjindal/Desktop/covid.pkl", map_location=torch.device('cpu'))  # Replace 'path_to_your_model.pkl' with the path to your model file
model.eval()

def classify_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted_idx = torch.max(outputs, 1)
    predicted_label = custom_labels[predicted_idx.item()]
    return predicted_label

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = classify_image(contents)
    return {"prediction": prediction}

@app.get("/")
async def main():
    content = """
    <html>
    <head>
        <title>Image Classification</title>
    </head>
    <body>
        <h1>Image Classification</h1>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

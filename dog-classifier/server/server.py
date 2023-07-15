from flask import Flask, request, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
from VGG16 import VGG16
from doodles import *
import torchvision.transforms as transforms
import io

app = Flask(__name__)
CORS(app)

classes = ['Maltese', 'Shih-Tzu', 'Beagle', 'Golden Retriever', 'Border Collie', 
           'Rottweiler', 'Great Dane', 'Husky', 'Pug', 'Samoyed']

model = VGG16(10)
model.to(torch.device('cpu'))
model.load_state_dict(torch.load('trained_10.pth', map_location=torch.device('cpu')), strict=False)

#test lol
@app.route('/')
def home():
    return 'helo'

@app.route('/predict', methods=['POST'])
def predict_image():
    image_file = request.files.get('image')
    if image_file is not None:
        dog = request.files['image'] 
        dog_data = dog.read()
        file = io.BytesIO(dog_data)
        pil = Image.open(file).convert('RGB')

        result = run_model(pil)

        drawing_filename = f'{result}.png'
        drawing_url = f'doodles/{drawing_filename}'
        return {"prediction": result, 'doodle_url': drawing_url}
    else:
        return {"prediction": "NOOOOO"}
    
def run_model(image):
    #Transforms from testing
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(size=(224, 224), antialias=True),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],)])
    
    transformed = transform(image).float()
    tensor = transformed.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(tensor).argmax()
    breed = classes[output.item()]
    print(breed)
    
    return breed

#Get doodles locally
@app.route('/doodles/<path:breed>', methods=['GET'])
def get_doodle(breed):
    folder = 'doodles'  
    return send_from_directory(folder, breed)

if __name__ == '__main__':
    app.run()

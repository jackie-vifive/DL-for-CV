from flask import Flask, render_template, request, send_file,send_from_directory, jsonify, url_for
import os
import werkzeug
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory where uploaded files will be saved
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 16 MB max upload limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    model_selection = request.form['model_selection']

    if model_selection == '1':
        model = ColorizationModel1()  # Assuming this class is defined
        model_file = 'colorization_model_1.pth'
    elif model_selection == '2':
        model = ColorizationModel2()  # Assuming this class is defined
        model_file = 'colorization_model_2.pth'
    elif model_selection == '3':
        model = ColorizationModel2()  # Assuming this class is defined
        model_file = 'colorization_model_2.pth'



    print("upload")
    file = request.files['file']
    if file:
        filename = werkzeug.utils.secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return process_video(filename, model, model_file)  # Function to process the video
    return 'No file uploaded'

def process_video(filename, model, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    model.to(device)

    print(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Path to the uploaded video
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Open the black and white video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video path
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Resize to the input size that your model expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Use the same mean and std used during training
    ])

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale if it's not already
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_pil = transform(gray_frame).unsqueeze(0).to(device)

        # Colorize the frame using the model
        with torch.no_grad():
            output_tensor = model(gray_frame_pil)

        # Reverse the preprocessing steps
        output_tensor = output_tensor.squeeze(0).cpu().detach()
        output_tensor = output_tensor * 0.5 + 0.5  # Reverse the normalization
        output_image = transforms.ToPILImage()(output_tensor)

        # Resize the output image to the original dimensions
        output_image = output_image.resize((frame_width, frame_height), Image.BILINEAR)

        # Convert PIL image back to OpenCV format and write to the video
        output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        out.write(output_image)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Return the path of the processed video
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_' + filename, as_attachment=True)


class ColorizationModel1(nn.Module):
    def __init__(self):
        super(ColorizationModel1, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, grayscale_image):
        # Encode the input grayscale image
        encoded = self.encoder(grayscale_image)

        # Decode to obtain the colorized image
        colorized_image = self.decoder(encoded)

        return colorized_image

class ColorizationModel2(nn.Module):
    def __init__(self):
        super(ColorizationModel2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, grayscale_image):
        # Encode the input grayscale image
        encoded = self.encoder(grayscale_image)

        # Bottleneck
        bottleneck = self.bottleneck(encoded)

        # Decode to obtain the colorized image
        colorized_image = self.decoder(bottleneck)

        return colorized_image

class ColorizationModel3(nn.Module):
    def __init__(self):
        super(ColorizationModel3, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, grayscale_image):
        # Encode the input grayscale image
        encoded = self.encoder(grayscale_image)

        # Bottleneck
        bottleneck = self.bottleneck(encoded)

        # Decode to obtain the colorized image
        colorized_image = self.decoder(bottleneck)

        return colorized_image

if __name__ == '__main__':
    app.run(debug=True, port=5001)

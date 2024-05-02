import os
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import CNN

# Load the dataset (assuming images are in a directory named 'static/uploads')
dataset_path = 'static/uploads'
image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load the model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

def preprocess_and_display_images(image_files):
    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        # Preprocess the image
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        # Make prediction
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        # Display information
        title = disease_info['disease_name'][index]
        description = disease_info['description'][index]
        prevent = disease_info['Possible Steps'][index]
        print("Image:", image_file)
        print("Disease:", title)
        print("Description:", description)
        print("Preventive Steps:", prevent)
        print("Image Path:", image_path)
        print("-" * 50)

# Preprocess and display images
preprocess_and_display_images(image_files)

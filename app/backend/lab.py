import torchvision.models as models
import torch.nn as nn
import torch
from torchinfo import summary
from PIL import Image
from torchvision import transforms

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load('app/ai_models/cats-dogs/cats-dogs-resnet50.pth', map_location=torch.device(device)))

# Creating labels
labels = ['cat', 'dog']

# Print model summary
# print(summary(model, input_size=(1, 3, 150, 150)))

# Getting image path
image_path = input("Enter the image path: ")

# Loading image as PIL image
img = Image.open(image_path)


# Print image
# img.show()

# Transforming image
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = simple_transform(img)

# Adding batch dimension
img = img.unsqueeze(0)

# Making prediction
model.eval()
with torch.inference_mode():
    # Setting device
    img = img.to(device)
    model = model.to(device)

    # Forward pass
    output = model(img)

    # Getting probabilities
    img_probs = torch.nn.functional.softmax(output, dim=1)

    # Getting highest probability class
    pred = torch.argmax(img_probs, dim=1)

    # Getting class label
    label = 'cat' if pred == 0 else 'dog'

    print(f"Predicted class is: {label}")
    print(f"Probability of class: {img_probs[0][pred].item() * 100:.2f}%")


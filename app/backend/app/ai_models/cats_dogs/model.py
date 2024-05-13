import torchvision.models as models
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms

class CatsDogsModel():
    def __init__(self):
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load('app/ai_models/cats_dogs/cats-dogs-resnet50.pth'))
        model.eval()
        self.model = model
        self.labels = ['cat', 'dog']
        print('Model loaded successfully')


    def predict(self, img):
        # Opening image
        img = Image.open(img)

        # Transforming image
        simple_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        img = simple_transform(img)

        # Adding batch dimension
        img = img.unsqueeze(0)

        # Making prediction
        with torch.inference_mode():
            # Forward pass
            output = self.model(img)

            # Getting probabilities
            img_probs = torch.nn.functional.softmax(output, dim=1)

            # Getting highest probability class
            pred = torch.argmax(img_probs, dim=1)

            # Getting class label
            label = self.labels[pred]

            # Setting up return dict
            return_dict = {
                'label': label,
                'probabilities': img_probs[0][pred].item(),
            }

            return return_dict
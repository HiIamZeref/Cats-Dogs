from flask import Flask, request, jsonify
from ai_models.cats_dogs.model import CatsDogsModel
from PIL import Image
from flask_cors import CORS

# Instantiating model
model = CatsDogsModel()

app = Flask(__name__)
CORS(app)


@app.route('/make_predictions', methods=['POST'])
def make_predictions():
    # Check if image exists in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'})
    
    # Getting image
    img = request.files['image']

    # Making predictions
    predictions = model.predict(img)

    # Printing predictions
    print(predictions)

    # Returning predictions
    return jsonify(predictions)

    
    

    

if __name__ == '__main__':
    app.run(debug=True)
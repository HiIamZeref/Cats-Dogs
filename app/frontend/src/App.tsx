import { useState } from "react";
import "./App.css";
import { makePrediction } from "./services/PredictionsApi";

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const onImageChange = (event) => {
    const selectedImage = event.target.files[0];
    setImage(selectedImage);
    setImageUrl(URL.createObjectURL(selectedImage));
  };

  const handleFileUpload = () => {
    const formData = new FormData();
    formData.append("image", image);

    makePrediction(formData)
      .then((response) => {
        const responseData = response.data;

        setPrediction(responseData);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  return (
    <>
      <div>
        <h1>Cats & Dogs App!</h1>
        <h2>
          This is a simple app that predicts if the image is a cat or a dog.
        </h2>
        <input type="file" accept="image/*" onChange={onImageChange} />
        <button onClick={handleFileUpload}>Make prediction!</button>
        {imageUrl && (
          <img
            src={imageUrl}
            alt="Uploaded"
            style={{ maxWidth: "300px", maxHeight: "300px" }}
          />
        )}
        {prediction && (
          <div>
            <h4>Prediction: {prediction.label}</h4>
            <p>Probability: {(prediction.probabilities * 100).toFixed(2)}%</p>
          </div>
        )}
      </div>
    </>
  );
}

export default App;

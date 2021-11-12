import uvicorn
from fastapi import FastAPI, UploadFile, File
from model import read_image, create_data_batch, make_prediction, get_prediction_label, load_model

# Import model
model = load_model("20211017-17281634491703-1000-images-mobilenetv2-Adam.h5")

app = FastAPI()

@app.get("/api")
async def read_root():
  return {"message": "Welcome to BikeHunt API"}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
  extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
  if not extension:
    return "Image must be jpg or png format!"
  # Read the file uploaded by the user
  image = read_image(await file.read())
  # Apply preprocessing
  batched_image = create_data_batch(image)
  # Make prediction
  prediction = make_prediction(batched_image, model)
  # Get prediction label
  label = get_prediction_label(prediction)
  return {"prediction": label}

if __name__ == '__main__':
  uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
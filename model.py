import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
import numpy as np

def read_image(image_encoded):
  pil_image = Image.open(BytesIO(image_encoded))
  return pil_image

def load_model(model_path):
  model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def process_image(uploaded_image, img_size=224):
  image = np.asarray(uploaded_image.resize((img_size, img_size)))[..., :3]
  image = np.expand_dims(image, 0)
  image = image / 127.5 - 1.0
  return image

def create_data_batch(image: Image.Image, batch_size=32):
  image = process_image(image)
  data = tf.data.Dataset.from_tensor_slices((tf.constant(image)))
  data_batch = data.batch(batch_size)
  return data_batch

def make_prediction(image, model):
  return model.predict(image)

def get_prediction_label(prediction):
  return ['Mountain Bike', 'Road Bike'][np.argmax(prediction)]
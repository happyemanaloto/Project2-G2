
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cifar10_transfer_model.keras")
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
IMG_SIZE = 160

def predict(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return {class_names[i]: float(preds[i]) for i in range(10)}

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Classifier",
    description="Upload an image to get top-3 predictions from the MobileNetV2 model fine-tuned on CIFAR-10."
)

app.launch()

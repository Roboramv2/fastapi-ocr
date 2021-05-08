from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import numpy as np
from cv2 import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()
model = load_model('digitsocr.h5')

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def pred(img):
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = cv2.resize(img,(32,32))
    img = np.array(img)
    try:
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
    except:
        pass
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img) 
    img = img.reshape(1, 32, 32, 1)
    classindex = int(model.predict_classes(img))
    return classindex

@app.post("/scorefile/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = pred(image)
    return prediction
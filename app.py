from flask import Flask
import os 
from utils import createModel
import numpy as np 
import cv2 

model = createModel()
model.load_weights('bestModel.h5')
predictionDict = {
    0:'With Covid',
    1:'Without Covid'
}

def predictNewImage(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(224,224))
    img = np.expand_dims(img, axis=0)
    pred = np.argmax(model.predict(img),axis=1)[0]
    return predictionDict[pred]

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'

if __name__ == '__main__':
    app.run()
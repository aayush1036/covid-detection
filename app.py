from flask import Flask
import os 
from utils import createModel,predictNew
model = createModel()
model.load_weights('bestModel.h5')
predictionDict = {
    0:'With Covid',
    1:'Without Covid'
}

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'

if __name__ == '__main__':
    app.run()
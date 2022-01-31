from flask import Flask,render_template,url_for
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
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
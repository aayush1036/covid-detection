from flask import Flask,render_template,url_for,request
import os
from utils import createModel,predictNew
model = createModel()
model.load_weights('bestModel.h5')
predictionDict = {
    0:'With Covid',
    1:'Without Covid'
}

app = Flask(__name__)
uploadPath = 'static/uploads'
if not os.path.exists(uploadPath):
    os.makedirs(uploadPath)

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/GetData', methods=['GET'])
def getData():
    return render_template('getData.html')

@app.route('/GetData',methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            imgPath = os.path.join(uploadPath,image.filename)
            image.save(imgPath)
            pred = predictNew(model=model, filepath=imgPath, labelDict=predictionDict)
            print(pred)
            return render_template('getData.html',pred=pred,imgPath=imgPath)
            

if __name__ == '__main__':
    app.run(debug=True)
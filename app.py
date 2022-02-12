from flask import Flask,render_template,url_for,request,redirect
import os
from deployUtils import createModel,predictNew
model = createModel()
model.load_weights('bestModel.h5')
predictionDict = {
    0:'With Covid',
    1:'Without Covid'
}
allowedFileTypes = ['jpg','png','jpeg']
app = Flask(__name__)
uploadPath = 'static/uploads'
if not os.path.exists(uploadPath):
    os.makedirs(uploadPath)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/GetData', methods=['GET'])
def getData():
    return render_template('getData.html')

@app.route('/GetData',methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            if image.filename == '':
                statusCode = 'NoFile'
                return render_template('getData.html',statusCode=statusCode, pred=None)
            if image.filename.rsplit('.',1)[1] not in allowedFileTypes:
                statusCode = 'InvalidFileType'
                return render_template('getData.html',statusCode=statusCode, pred=None)
            else:
                statusCode = 'Success'
                imgPath = os.path.join(uploadPath,image.filename)
                image.save(imgPath)
                pred = predictNew(model=model, filepath=imgPath, labelDict=predictionDict,surety=True)
                return render_template('getData.html',statusCode=statusCode,pred=pred)
            

if __name__ == '__main__':
    app.run(debug=True)
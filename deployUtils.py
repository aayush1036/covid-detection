from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

def createModel()->Sequential:
    """Creates a model 

    Returns:
        tf.keras.models.Sequential: The created model
    """    
    model = Sequential(layers=[
        Conv2D(filters=32, kernel_size=(3,3),activation='relu',input_shape=(224,224,3)),
        MaxPooling2D(),
        Conv2D(filters=64, kernel_size=(3,3),activation='relu'),
        MaxPooling2D(),
        Dropout(rate=0.3),
        Flatten(),
        Dense(units=128,activation='relu'),
        Dense(units=2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def predictNew(model:Sequential,filepath:str,targetSize=(224,224),labelDict=None,surety=False)->str:
    """Predict a new x-ray

    Args:
        model (tf.keras.models.Sequential): The model which you would want to use to make predictions
        filepath (str): The file path of the image you want to predict
        targetSize (tuple, optional): The input size of the image for your model. Defaults to (224,224).
        labelDict (dict, optional): The dictionary containing id as keys and label as values. Defaults to None.

    Returns:
        str: Predtcion made by the model
    """    
    img = load_img(path=filepath, target_size=targetSize)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    predIdx = np.argmax(pred, axis=1)[0]
    if surety and labelDict:
        confidence = pred[0][predIdx]
        message = f'The model predicted {labelDict[predIdx]} with {confidence:.3%} confidence'
        return message
    if labelDict:
        return labelDict[predIdx]
    else:
        return predIdx
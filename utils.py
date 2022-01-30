import os 
import shutil
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import cv2 

def isMoved():
    if os.path.exists('Dataset/'):
        return True 
    else:
        return False 

def moveFiles(
    BASE_PATH='archive/DATA/DATA',
    LABEL_PATH='archive/labels_Covid.csv', 
    DESTINATION='Dataset',
    TEST_SIZE=0.2,
    VAL_SIZE = 0.1,
    RANDOM_STATE=42):

    labelFile = pd.read_csv(LABEL_PATH)
    classId = labelFile['ClassId'].unique()
    labels = labelFile['Name'].unique()
    labelDict = dict(zip(classId, labels))

    X = []
    y = []
    categories = os.listdir(BASE_PATH)
    for category in categories:
        files = os.listdir(os.path.join(BASE_PATH,category))
        for file in files:
            X.append(file)
            y.append(category)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train,y_val = train_test_split(
        X_train,y_train, test_size=VAL_SIZE, stratify=y_train,random_state=RANDOM_STATE)

    for label in labelDict.values(): 
        trainPath = os.path.join(DESTINATION,'Train',label)
        valPath = os.path.join(DESTINATION,'Val',label)
        testPath = os.path.join(DESTINATION,'Test',label)
        if not os.path.exists(trainPath):
            os.makedirs(trainPath)
        if not os.path.exists(valPath):
            os.makedirs(valPath)
        if not os.path.exists(testPath):
            os.makedirs(testPath)

    for fileName, label in zip(X_train, y_train):
        shutil.move(
            src=os.path.join(BASE_PATH, label,fileName),
            dst=os.path.join(DESTINATION,'Train',labelDict[int(label)],fileName)
        )
    for fileName, label in zip(X_val, y_val):
        shutil.move(
            src=os.path.join(BASE_PATH, label,fileName),
            dst=os.path.join(DESTINATION,'Val',labelDict[int(label)],fileName)
        )
    
    for fileName, label in zip(X_test, y_test):
        shutil.move(
            src=os.path.join(BASE_PATH, label,fileName),
            dst=os.path.join(DESTINATION,'Test',labelDict[int(label)],fileName)
        )
    numTrain = 0
    numVal = 0 
    numTest = 0 

    for category in labelDict.values():
        numTrain += len(os.listdir(os.path.join(DESTINATION,'Train',category)))
        numVal += len(os.listdir(os.path.join(DESTINATION,'Val',category)))
        numTest += len(os.listdir(os.path.join(DESTINATION,'Test',category)))

    assert numTrain == len(X_train) and numVal == len(X_val) and numTest == len(X_test)
    shutil.rmtree('archive/')  
    print('Created folders successfully')

def seeExamples(
    generator,
    figsize=(16,16),
    nrows=4,
    ncols=4,
    labelDict = None):
    if labelDict is None:
        labelDict = dict(zip(generator.class_indices.values(), generator.class_indices.keys()))
    numBatches = len(generator)
    indices = np.random.randint(low=0,high=numBatches, size=(nrows,ncols))      
    fig, ax = plt.subplots(figsize=figsize,nrows=nrows,ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            images, labels = generator[indices[i,j]]
            numExamples = len(images)
            exampleIdx = np.random.randint(low=0, high=numExamples)
            image = images[exampleIdx]
            label = labels[exampleIdx]
            label = np.where(label == 1)[0][0]
            ax[i,j].axis('off')
            ax[i,j].imshow(image)
            ax[i,j].set_title(labelDict[label])
    plt.tight_layout()
    plt.show()

def plotHistory(
    history,
    figsize=(16,6),
    accTitle = 'Accuracy over epochs',
    accXlabel='Epochs',
    accYlabel='Accuracy',
    lossTitle='Loss over epochs',
    lossXlabel='Epochs',
    lossYlabel='Loss',
    accPosition=0,
    lossPosition=1,
    save=True):
    
    history = pd.DataFrame(history.history)
    fig, ax = plt.subplots(figsize=figsize, nrows=1,ncols=2)
    history[['accuracy','val_accuracy']].plot(ax=ax[accPosition],title=accTitle,xlabel=accXlabel,ylabel=accYlabel)
    history[['loss','val_loss']].plot(ax=ax[lossPosition],title=lossTitle,xlabel=lossXlabel,ylabel=lossYlabel)
    plt.show()
    if save:
        if not os.path.exists('Images/'):
            os.makedirs('Images/')
        savePath = 'Images/history.png'
        plt.savefig(savePath)



def evaluateModel(model,trainSet,validationSet,testSet):
    _, trainAcc = model.evaluate(trainSet)
    _, valAcc = model.evaluate(validationSet)
    _, testAcc = model.evaluate(testSet)

    print(f'The accuracy on train set is {trainAcc:.3%}')
    print(f'The accuracy on validation set is {valAcc:.3%}')
    print(f'The accuracy on test set is {testAcc:.3%}')

def predictNew(model,filepath,targetSize=(224,224),labelDict=None):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=targetSize)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)[0]
    if labelDict is not None:
        return labelDict[pred]
    else:
        return pred
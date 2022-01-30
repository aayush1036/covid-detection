import os 
import shutil
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import cv2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
plt.style.use('seaborn')

def isMoved():
    """Checks if the data is already moved to train, validate and test folders

    Returns:
        Boolean: A boolean value signifying whether the data is moved or not
    """    
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
    """Moves the data to train, validate and test folders

    Args:
        BASE_PATH (str, optional): The path where the data has been extracted from Kaggle. Defaults to 'archive/DATA/DATA'.
        LABEL_PATH (str, optional): The path where the labels are stored. Defaults to 'archive/labels_Covid.csv'.
        DESTINATION (str, optional): The path where you would like to move the data to. Defaults to 'Dataset'.
        TEST_SIZE (float, optional): The proportion of data you want to retain for the test set. Defaults to 0.2.
        VAL_SIZE (float, optional): The proportion of training data you want to retain for validation set. Defaults to 0.1.
        RANDOM_STATE (int, optional): The seed you want while splitting the data. Defaults to 42.
    Returns:
        None
    """    

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
    """Plot the examples from the generator

    Args:
        generator (DirectoryIterator): The generator you want to see the examples from
        figsize (tuple, optional): The size of matplotlib figure. Defaults to (16,16).
        nrows (int, optional): The number of rows in the figure. Defaults to 4.
        ncols (int, optional): The number of columns in the figure. Defaults to 4.
        labelDict (dict, optional): The dictionary containing id as keys and labels as values. Defaults to None.
    Returns:
        None
    """    
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
    """Plots the history for better visualization of training

    Args:
        history (dict): The dictionary containing accuracy and loss for the epochs
        figsize (tuple, optional): Size of figure. Defaults to (16,6).
        accTitle (str, optional): Title for accuracy chart. Defaults to 'Accuracy over epochs'.
        accXlabel (str, optional): xlabel for accuracy chart. Defaults to 'Epochs'.
        accYlabel (str, optional): ylabel for accuracy chart. Defaults to 'Accuracy'.
        lossTitle (str, optional): Title for loss chart. Defaults to 'Loss over epochs'.
        lossXlabel (str, optional): xlabel for loss chart. Defaults to 'Epochs'.
        lossYlabel (str, optional): ylabel for loss chart. Defaults to 'Loss'.
        accPosition (int, optional): Position where you would want to plot accuracy. Defaults to 0.
        lossPosition (int, optional): Position where you would want to plot loss. Defaults to 1.
        save (bool, optional): Saves the chart if set to True. Defaults to True.
    Returns:
        None
    """    
    
    history = pd.DataFrame(history.history)
    fig, ax = plt.subplots(figsize=figsize, nrows=1,ncols=2)
    history[['accuracy','val_accuracy']].plot(ax=ax[accPosition],title=accTitle,xlabel=accXlabel,ylabel=accYlabel)
    history[['loss','val_loss']].plot(ax=ax[lossPosition],title=lossTitle,xlabel=lossXlabel,ylabel=lossYlabel)
    if save:
        if not os.path.exists('Images/'):
            os.makedirs('Images/')
        savePath = 'Images/history.png'
        plt.savefig(savePath)
    plt.show()



def evaluateModel(model,trainSet,validationSet,testSet):
    """Evaluate the model on train, validation and test set and print accuracy

    Args:
        model (tf.keras.models.Sequential): The model which you would want to evaluate
        trainSet (DirectoryIterator): The training generator
        validationSet (DirectoryIterator): The validation generator
        testSet (DirectoryIterator): The test generator
    Returns:
        None
    """    
    _, trainAcc = model.evaluate(trainSet)
    _, valAcc = model.evaluate(validationSet)
    _, testAcc = model.evaluate(testSet)

    print(f'The accuracy on train set is {trainAcc:.3%}')
    print(f'The accuracy on validation set is {valAcc:.3%}')
    print(f'The accuracy on test set is {testAcc:.3%}')

def predictNew(model,filepath,targetSize=(224,224),labelDict=None):
    """Predict a new x-ray

    Args:
        model (tf.keras.models.Sequential): The model which you would want to use to make predictions
        filepath (str): The file path of the image you want to predict
        targetSize (tuple, optional): The input size of the image for your model. Defaults to (224,224).
        labelDict (dict, optional): The dictionary containing id as keys and label as values. Defaults to None.

    Returns:
        str: Predtcion made by the model
    """    
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

def plotProportions(trainSet,validationSet,testSet,figsize=(12,4),
    trainPos = 0,
    validationPos = 1,
    testPos = 2,
    trainTitle='Proportion of cases in train set',
    validationTitle='Proportion of cases in validation set',
    testTitle='Porportion of cases in test set',
    precision=0,
    save=True):
    """Plots the proportion of covid positive and negative cases

    Args:
        trainSet (DirectoryIterator): The training set for the model
        validationSet (DirectoryIterator): The validtion set for the model
        testSet (DirectoryIterator): The test set for the model
        figsize (tuple, optional): The size of the figure. Defaults to (12,4).
        trainPos (int, optional): The position where you want to plot the proportion of train set. Defaults to 0.
        validationPos (int, optional): The position where you want to plot the proportion of test set. Defaults to 1.
        testPos (int, optional): The position where you want to plot the proportion of test set. Defaults to 2.
        trainTitle (str, optional): The title you want to give to the chart of training set. Defaults to 'Proportion of cases in train set'.
        validationTitle (str, optional): The title you want to give to the chart of vallidation set. Defaults to 'Proportion of cases in validation set'.
        testTitle (str, optional): The title you want to give to the chart of test set. Defaults to 'Porportion of cases in test set'.
        precision (int, optional): The precision of percentage you want on the chart. Defaults to 0.
        save (bool, optional): Saves the chart if set to True. Defaults to True.
    Returns:
        None
    """    
    trainProportions = pd.Series(trainSet.labels).value_counts()
    validationProportions = pd.Series(validationSet.labels).value_counts()
    testProportions = pd.Series(testSet.labels).value_counts()
    fig, ax = plt.subplots(figsize=figsize,nrows=1,ncols=3)
    trainProportions.plot(kind='pie', ax = ax[trainPos],title=trainTitle,autopct=f'%1.{precision}f%%')
    validationProportions.plot(kind='pie',ax=ax[validationPos],title=validationTitle,autopct=f'%1.{precision}f%%')
    testProportions.plot(kind='pie',ax=ax[testPos],title=testTitle,autopct=f'%1.{precision}f%%')
    if save:
        if not os.path.exists('Images/'):
            os.makedirs('Images/')
        plt.savefig('Images/Proportions.png')
    plt.show()

def createModel():
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
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,MaxPool2D,BatchNormalization
import keras
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

train_copy = train.copy()
test_copy = test.copy()

y_train=train['label']
x_train=train.drop(['label'],axis=1).values
y_test=test['label']
x_test=test.drop(['label'],axis=1).values

x_train = x_train / 255
x_test = x_test / 255
x_train =x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

y_train_vals=train['label']
x_train_vals=train.drop(['label'],axis=1).values
y_test_vals=test['label']
x_test_vals=test.drop(['label'],axis=1).values

x_train_vals = x_train_vals / 255
x_test_vals = x_test_vals / 255
x_train_vals =x_train_vals.reshape(-1,28,28,1)
x_test_vals = x_test_vals.reshape(-1,28,28,1)


plt.imshow(x_train[0].reshape(28,28))

plt.figure(figsize=(10,10))
for i in range(40):
    plt.subplot(6,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

label_bin = LabelBinarizer()
labels_train = label_bin.fit_transform(y_train)
labels_test = label_bin.fit_transform(y_test)
x_trains, x_tests, y_trains, y_tests = train_test_split(x_train, labels_train, test_size = 0.3)

generate = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)
generate.fit(x_trains)

model=Sequential()
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(24,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

mod_hist = model.fit(generate.flow(x_train,y_train, batch_size = 128),validation_data=(x_tests,y_tests),epochs=20,batch_size=64)

plt.plot(mod_hist.history['acc'])
plt.plot(mod_hist.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()

y_pred = model.predict(x_tests)
y_vals_te = np.argmax(y_pred,axis=1)

def graph_function():
    plt.figure(figsize=(12,8))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_tests[i],cmap='gray')
        plt.xlabel(f"Actual: {y_vals_te[i]}\n Predicted: {y_pred[i]}")
    plt.tight_layout()
    plt.show()

graph_function()


x_training, x_testing, y_training, y_testing = train_test_split(x_train_vals, labels_train, test_size = 0.3)

def bestParametersLogistic():
    log = LogisticRegression()
    parameters = [{
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
    }]
    log_search = GridSearchCV(log,parameters,cv=5,scoring='accuracy')
    log_search.fit(x_training,y_training)
    print(log_search.best_params_)
    
log = LogisticRegression()
log.fit(x_training,y_training)

log_pred = log.predict(x_testing)

def logistic():
    logist = LogisticRegression(penalty='l1', solver='liblinear', C=1)
    logist.fit(x_training,y_training)
    log_predict = logist.predict(x_testing)
    log_score = accuracy_score(y_testing,log_predict)*100  
    print(log_predict)
    print(log_score)
    
    #Use the ROC/AUC curve to find the Score 
    log_prob = logist.predict_proba(x_testing)[:,1]
    false_positive_log,true_positive_log,threshold_log = roc_curve(y_testing,log_prob)
    auc_score = metrics.auc(false_positive_log,true_positive_log)
    print("AUC:", auc_score)
    plt.plot(false_positive_log,true_positive_log)
    plt.plot([0,1], ls='--')

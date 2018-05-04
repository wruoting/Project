import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from preprocessing import *
import matplotlib.pyplot as plt

close_data,close_date = load_ticker_close('MSFT')
AAPL_close_data,AAPL_close_date = load_ticker_close('AAPL')
TRAIN_SIZE = 20 #for 20 SMA
PREDICT_TIME = 1
LAG_TIME = 1

#This is our training data, we will scale it
X_train,X_test,Y_train,Y_test = split_into_chunks(close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME,binary = False,scale = False,percent = 0.9)


X_trainp, X_testp, Y_trainp, Y_testp  = split_into_chunks(close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME, binary=False,scale= False,percent = 0.9)

model = Sequential()
model.add(Dense(750,input_shape = (TRAIN_SIZE,)))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dense(75))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer = 'adam', loss ='mse',metrics = ['mse'])
model.summary()
model.fit(
    X_train,
    Y_train,
    epochs = 50,
    batch_size=75,
    verbose=1,
    validation_split=0.1
    )
score,acc = model.evaluate(X_test,Y_test,batch_size = 75)

print(score,acc)

predicted = model.predict(X_test)

#scale by x_testp (the real data)


try:
    fig = plt.figure()
    #plt.plot(Y_test[:150], color='black') # BLACK - trained RESULT
    #plt.plot(predicted[:150], color='blue') # BLUE - trained PREDICTION
    plt.plot(Y_testp[:150], color='green',label = "Actual Result") # GREEN - actual RESULT
    plt.plot(predicted[:150], color='red',label = "Predicted Result") # RED - restored PREDICTION
    plt.xlabel('Data Point')
    plt.ylabel('Closing price ($)')
    plt.legend()
    plt.show()
except Exception as e:
    print(e)


X_trainp_AAPL, X_testp_AAPL, Y_trainp_AAPL, Y_testp_AAPL  = split_into_chunks(AAPL_close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME, binary=False,scale= False,percent = 0.9)

predicted_AAPL = model.predict(X_testp_AAPL)
try:
    fig = plt.figure()
    #plt.plot(Y_test[:150], color='black') # BLACK - trained RESULT
    #plt.plot(predicted[:150], color='blue') # BLUE - trained PREDICTION
    plt.plot(Y_testp_AAPL[:150], color='green',label = "Actual Result") # GREEN - actual RESULT
    plt.plot(predicted_AAPL[:150], color='red',label = "Predicted Result") # RED - restored PREDICTION
    plt.xlabel('Data Point')
    plt.ylabel('Closing price ($)')
    plt.legend()
    plt.show()
except Exception as e:
    print(e)
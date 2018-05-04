import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from preprocessing import *
import matplotlib.pyplot as plt

close_data,close_date = load_ticker_close('MSFT')
TRAIN_SIZE = 20 #for 20 SMA
PREDICT_TIME = 1
LAG_TIME = 1

#This is our training data, we will scale it
X_train,X_test,Y_train,Y_test = split_into_chunks(close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME,binary = True,scale = False,percent = 0.9)

X_trainp, X_testp, Y_trainp, Y_testp  = split_into_chunks(close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME, binary=True,scale= False,percent = 0.9)


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
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer = 'adam', loss ='binary_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit(
    X_train,
    Y_train,
    epochs = 60,
    batch_size=75,
    verbose=1,
    validation_split=0.1
    )
score,acc = model.evaluate(X_test,Y_test,batch_size = 75)

print(score,acc)

# params = []
# for xt in X_test:
#     xt = np.array(xt)
#     mean_ = xt.mean()
#     scale_ = xt.std()
#     params.append([mean_, scale_])
# new_predicted = []
# for pred, par in zip(predicted, params):
#     a = pred*par[1]
#     a += par[0]
#     new_predicted.append(a)
predicted = model.predict(X_test)

print(predicted)
#scale by x_testp (the real data)


# mse = mean_squared_error(predicted, new_predicted)
# print(mse)

try:
    fig = plt.figure()
    #plt.plot(Y_test[:150], color='black') # BLACK - trained RESULT
    #plt.plot(predicted[:150], color='blue') # BLUE - trained PREDICTION
    plt.plot(Y_testp[:150,0] * 100, color='green',label = "Actual Result") # GREEN - actual RESULT
    plt.plot(predicted[:150,0] * 100, color='red',label = "Predicted Result") # RED - restored PREDICTION
    plt.xlabel('Data Point')
    plt.ylabel('Prediction Percentage (%)')
    plt.legend()
    plt.show()
except Exception as e:
    print(e)

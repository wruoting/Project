import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.Session() as sess:
    print (sess.run(c))
#
# import keras
# from keras.models import Sequential
# from keras.layers import Dense,Dropout,Activation
# from keras.optimizers import SGD
# from preprocessing import *
#
# close_data,close_date = load_ticker_close('AAPL')
# TRAIN_SIZE = 20 #for 20 SMA
# PREDICT_TIME = 1
# LAG_TIME = 1
#
# X,Y = split_into_chunks(close_data,TRAIN_SIZE,PREDICT_TIME,LAG_TIME,binary = False, scale = True, test = False)
# X,Y = np.array(X),np.array(Y)
#
# #create our training/testing data
# X_train,X_test,Y_train,Y_test = create_Xt_Yt(X,Y,percent = 0.9)
#
# model = Sequential()
# model.add(Dense(500,input_shape = (TRAIN_SIZE,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(Dense(250))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('linear'))
# model.compile(optimizer = 'adam', loss ='mse')
#
# model.fit(
#     X_train,
#     Y_train,
#     nb_epoch = 5,
#     batch_size=128,
#     verbose=1,
#     validation_split=0.1
#     )
# score = model.evaluate(X_test,Y_test,batch_size = 128)
#
# print(score)

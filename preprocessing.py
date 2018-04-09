from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import numpy as np

def load_ticker_close(ticker_name):
    f = open('./TickerData/'+ticker_name + '.csv','r').readlines()[1:]
    close_data = []
    close_date = []
    for line in f:
        close_data.append(float(line.split('\t')[4]))
        close_date.append(line.split('\t')[0])
    return close_data,close_date


#if this isnt a binary data set, we
#Data, training set, target time, lag size
def split_into_chunks(data,train,predict,step,binary = True, scale = True, test = False):
    X,Y = [],[]
    for i in range(0,len(data),step):
        try:
            if test:#something about the regression needs to be fixed for preprocessing
                x_i = data[i:i+train]
                y_i = data[i+train+predict]
                timeseries = np.array(data[i:i+train])
            else:
                x_i = data[i:i+train]
                y_i = data[i+train+predict]
                timeseries = np.array(data[i:i+train+predict])
            if binary:
                if y_i - data[i+train] > 0:
                    y_i =[1.,0.]
                else:
                    y_i = [0.,1.]

                if scale:
                    x_i = preprocessing.scale(x_i)
            else:
                if scale:
                    timeseries = preprocessing.scale(timeseries)
                x_i = timeseries[:-1]
                y_i = timeseries[-1]
        except Exception as e:
            print(e)
            break
        X.append(x_i)
        Y.append(y_i)
    return X, Y


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def create_Xt_Yt(X, Y, percent=0.8):
    X_index = int(np.round(len(X) * percent))
    Y_index = int(np.round(len(Y) * percent))

    X_train = X[0:X_index]
    Y_train = Y[0:Y_index]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[X_index:]
    Y_test = Y[Y_index:]

    return X_train, X_test, Y_train, Y_test

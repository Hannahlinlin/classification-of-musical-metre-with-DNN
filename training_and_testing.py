import numpy as np
import os
os.environ['KERAS_BACKEND']='theano'
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam, Adamax
#from keras import initializations
import scipy.io as sio
#import pydot

batch_size = 10
nb_epoch = 150
k_fold_value = 10   #全部資料分成幾組 200/10--test20+train180
data_size = 200
testing_size = int(data_size/k_fold_value)
training_size = int(data_size-testing_size)


# 讀取資料(還未分割training跟testing)
mat_contents = sio.loadmat('h_4class_data.mat')
all_data = mat_contents['b']

mat_contents = sio.loadmat('h_4class_target.mat')
all_target = mat_contents['c']

predictMatrix_all = np.zeros((data_size, 4))
loss_all = np.zeros(k_fold_value)
accuracy_all = np.zeros(k_fold_value)

for i in range(0, k_fold_value):  # 從1~10(11之前)
    x_test = np.zeros((int(testing_size), 54))
    y_test = np.zeros((int(testing_size), 4))
    x_train = np.zeros((int(training_size), 54))
    y_train = np.zeros((int(training_size), 4))
    for j in range(4):
        x_test[j * int(testing_size / 4):(j + 1) * int(testing_size / 4)][:] \
            = all_data[i * int(testing_size / 4) + j * int(data_size/4):(i + 1) * int(testing_size / 4) + j * int(data_size/4)]
        y_test[j * int(testing_size / 4):(j + 1) * int(testing_size / 4)][:]\
            = all_target[i * int(testing_size / 4) + j * int(data_size/4):(i + 1) * int(testing_size / 4) + j * int(data_size/4)]
        if i == 0:
            x_train[j * int(training_size/4):(j + 1) * int(training_size/4)][:] \
                = all_data[(i + 1) * int(testing_size/4) + j * int(data_size/4):(j + 1) * int(data_size/4)]
            y_train[j * int(training_size/4):(j + 1) * int(training_size/4)][:] \
                = all_target[(i + 1) * int(testing_size/4) + j * int(data_size/4):(j + 1) * int(data_size/4)]
        elif i == 9:
            x_train[j * int(training_size/4):(j + 1) * int(training_size/4)][:] \
                = all_data[j * int(data_size/4):j * int(data_size/4) + int(training_size/4)]
            y_train[j * int(training_size/4):(j + 1) * int(training_size/4)][:] \
                = all_target[j * int(data_size/4):j * int(data_size/4) + int(training_size/4)]
        else:
            x_train[j * int(training_size/4): j * int(training_size/4) + i * int(testing_size/4)][:] \
                = all_data[j * int(data_size/4): i * int(testing_size/4) + j * int(data_size/4)]
            y_train[j * int(training_size/4): j * int(training_size/4) + i * int(testing_size/4)][:] \
                = all_target[j * int(data_size/4): i * int(testing_size/4) + j * int(data_size/4)]
            x_train[j * int(training_size/4) + i * int(testing_size/4): (j + 1) * int(training_size/4)][:] \
                = all_data[j * int(data_size/4) + (i + 1) * int(testing_size/4): (j + 1) * int(data_size/4)]
            y_train[j * int(training_size/4) + i * int(testing_size/4): (j + 1) * int(training_size/4)][:] \
                = all_target[j * int(data_size/4) + (i + 1) * int(testing_size/4): (j + 1) * int(data_size/4)]

    # 建立模型
    model = Sequential([
        Dense(60, input_dim=54),
        # Dropout(0.25),
        Activation('relu'),

        Dense(60),
        Activation('relu'),

        Dense(4),
        Activation('softmax'),
    ])

    # Optimizers
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
    sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
    adamMax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )

    print('Training ------')
    model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)
    # print(model.history)         , validation_split =0.1

    print('\n Testing ------')
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('\n Predict ------')
    predictMatrix = model.predict(x_test, batch_size=batch_size, verbose=0)

    predictMatrix_all[i * testing_size: (i + 1) * testing_size][:] = predictMatrix
    loss_all[i] = loss
    accuracy_all[i] = accuracy

    print('test loss:', loss)
    print('test accuracy', accuracy)
    print('predictMatrix', predictMatrix)

print('test all loss:', loss_all)
print('test all accuracy', accuracy_all)
print('all predictMatrix', predictMatrix_all)


# 統計最後結果
result = np.zeros((4, 4))
for i in range(int(k_fold_value)):
    for j in range(4):
        for k in range(int(testing_size/4)):
            if j == 0:
                result[0][np.argmax(predictMatrix_all[i * int(testing_size) + j * int(testing_size/4) + k])] += 1
            elif j == 1:
                result[1][
                    np.argmax(predictMatrix_all[i * int(testing_size) + j * int(testing_size/4) + k])] += 1
            elif j == 2:
                result[2][
                    np.argmax(predictMatrix_all[i * int(testing_size) + j * int(testing_size/4) + k])] += 1
            else:
                result[3][
                    np.argmax(predictMatrix_all[i * int(testing_size) + j * int(testing_size/4) + k])] += 1

print(result)


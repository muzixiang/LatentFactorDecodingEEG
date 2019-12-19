from keras.layers  import Input, LSTM, Dense, Dropout
from keras.models import Model, Sequential
import h5py
import scipy.io as sio
import numpy as np
from sklearn import metrics

trialL=63*128
trialNum=40
latdim=16
step=128*3 #控制序列长度
method='1ae'
emodim=1 #valence 0 arousal 1
batch=40
epoch_num=5

def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x


#file = h5py.File('trial_labels_general_valence_arousal_dominance.mat','r')
file = h5py.File('trial_labels_personal_valence_arousal_dominance.mat','r')
l_data = file['trial_labels']
l_data = np.transpose(l_data)
print(l_data.shape)

pred_test_f1=[]
pred_train_f1=[]
pred_test_acc=[]
pred_train_acc=[]

for testSubNo in range(1,33):
    X_test = []
    X_train = []  # list类型
    if testSubNo==1:
        Y_test = l_data[0:testSubNo * trialNum, emodim]
        Y_train = l_data[testSubNo*trialNum:32*trialNum,emodim]
    elif testSubNo==32:
        Y_test = l_data[(testSubNo-1)*trialNum:testSubNo * trialNum, emodim]
        Y_train = l_data[0:(testSubNo-1)*trialNum, emodim]
    else:
        Y_test = l_data[(testSubNo - 1) * trialNum:testSubNo * trialNum, emodim]
        Y_train1 = l_data[0:(testSubNo - 1) * trialNum, emodim]
        Y_train2 = l_data[testSubNo * trialNum:32*trialNum, emodim]
        Y_train = np.concatenate((Y_train1,Y_train2))

    print('test subNo: '+str(testSubNo))
    file1 = sio.loadmat('D:\\VAE Experiment\\DEAP\\encoded_eegs_' + method + '\\encoded_eegs_'+ method +'_sub' + str(testSubNo) + '_latentdim' + str(latdim) + '.mat')

    testSubData = file1['encoded_eegs']
    testSubData = ZscoreNormalization(testSubData)
    #print(t_data.shape)
    #构造X_test序列集
    for trialNo in range(0,40):
        trial_data = testSubData[trialNo*trialL:(trialNo+1)*trialL,:]
        #以step等间隔采样
        trial_data = trial_data[0:trialL:step,:]
        #print(trial_data.shape)
        #将序列插入list，list中每个元素即一个序列
        X_test.append(trial_data)
    #print(len(X_test))
    #print(len(Y_test))

    # 构造X_train序列集
    for trainSubNo in range(1,33):
        if trainSubNo==testSubNo:
            continue
        file2 = sio.loadmat('D:\\VAE Experiment\\DEAP\\encoded_eegs_' + method + '\\encoded_eegs_'+ method + '_sub' + str(
            trainSubNo) + '_latentdim' + str(latdim) + '.mat')
        trainSubData = file2['encoded_eegs']
        trainSubData = ZscoreNormalization(trainSubData)
        for trialNo in range(0, 40):
            trial_data = trainSubData[trialNo * trialL:(trialNo + 1) * trialL, :]
            # 以step等间隔采样
            trial_data = trial_data[0:trialL:step, :]
            # 将序列插入list，list中每个元素即一个序列
            X_train.append(trial_data)
    print(len(X_train))
    print(len(Y_train))

    #构造RNN模型
    seqL = trial_data.shape[0]
    print('seqence length: '+str(seqL))
    model = Sequential()
    model.add(LSTM(200, input_shape=(seqL, latdim)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([X_train], Y_train, validation_data=([X_test], Y_test), epochs=epoch_num, batch_size=batch, shuffle=True)
    # predict on the training data after training...
    Y_test_predict = model.predict_classes([X_test])
    Y_train_predict = model.predict_classes([X_train])

    print(metrics.f1_score(Y_test, Y_test_predict))
    print(metrics.accuracy_score(Y_test, Y_test_predict))
    print(metrics.f1_score(Y_train, Y_train_predict))
    print(metrics.accuracy_score(Y_train, Y_train_predict))

    pred_test_f1.append(metrics.f1_score(Y_test, Y_test_predict))
    pred_test_acc.append(metrics.accuracy_score(Y_test, Y_test_predict))
    pred_train_f1.append(metrics.f1_score(Y_train, Y_train_predict))
    pred_train_acc.append(metrics.accuracy_score(Y_train, Y_train_predict))

if emodim==0:
    sio.savemat('lstm_personal_performance_ae_valence_subcross.mat',
            {'test_f1': pred_test_f1, 'test_acc': pred_test_acc, 'train_f1': pred_train_f1,
             'train_acc': pred_train_acc})
else:
    sio.savemat('lstm_personal_performance_ae_arousal_subcross.mat',
                {'test_f1': pred_test_f1, 'test_acc': pred_test_acc, 'train_f1': pred_train_f1,
                 'train_acc': pred_train_acc})
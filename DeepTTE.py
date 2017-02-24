from keras.optimizers import *
from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.backend import *
from keras.layers.wrappers import *
from keras.layers.advanced_activations import *
from keras.callbacks import *

from generator import *

import tensorflow as tf
import sys

NB_DRIVER = 14864
BATCH_SZ = 512
NB_EPOCH = 40

SAMPLE_RATE = 40

def seq_mape(y_true, y_pred):
    loss = K.abs((y_true - y_pred) / (10 +  y_true))
    return K.mean(loss, axis = 1) * 100.

class DeepTTE:
    def __init__(self):
        self.tensor = {}
        self.add_tensor()

    def add_tensor(self):
        self.tensor['driverID'] = Input(shape = (1, ), dtype = 'int32')
        self.tensor['weekID'] = Input(shape = (1, ), dtype = 'int32')
        self.tensor['timeID'] = Input(shape = (1, ), dtype = 'int32')
        self.tensor['dist'] = Input(shape = (1, ), dtype = 'float32')
        self.tensor['record'] = Input(shape = (SAMPLE_RATE, 5), dtype = 'float32')

    def build_model(self):
        # Attribute Component Begin
        driver = Embedding(input_dim = NB_DRIVER, output_dim = 16, input_length = 1)(self.tensor['driverID'])
        driver = Reshape((16, ))(driver)

        week = Embedding(input_dim = 7, output_dim = 3, input_length = 1)(self.tensor['weekID'])
        week = Reshape((3, ))(week)

        time = Embedding(input_dim = 1440, output_dim = 4, input_length = 1)(self.tensor['timeID'])
        time = Reshape((4, ))(time)

        dist = self.tensor['dist']

        attr = merge([driver, week, time, dist], mode = 'concat', concat_axis = 1)
        # Attribute Component End


        # Sequence Learning Component Start
        rep_attr = RepeatVector(SAMPLE_RATE)(attr)
        traj = merge([self.tensor['record'], rep_attr], mode = 'concat', concat_axis = 2)
        
        traj = LSTM(nb_units, return_sequences = True)(traj)
        traj = LSTM(nb_units, return_sequences = True)(traj)

        res = TimeDistributed(Dense(128, activation = lambda x: K.relu(x, alpha = 0.01)))(traj)
        res = TimeDistributed(Dense(64, activation = lambda x: K.relu(x, alpha = 0.01)))(traj)
        # Sequence Learning Component End


        # Auxiliary Component Start
        seq = TimeDistributed(Dense(1, activation = 'relu'), name = 'seq_output')(res)
        # Auxiliary Component End

        # Residual Component Start
        res = Reshape((SAMPLE_RATE * 64, ))(res)

        x = merge([driver, week, time, dist, res], mode = 'concat', concat_axis = 1)
        x = Dense(128, activation = lambda x: K.relu(x, alpha = 0.01))(x)

        for i in range(2):
            x = merge([x, Dense(128, activation = lambda x: K.relu(x, alpha = 0.01))(x)], mode = 'sum')
        # Residual Component End

        pred = Dense(1, activation = 'linear', name = 'pred_output')(x)

        tensors = map(lambda x: self.tensor[x], self.tensor.keys())

        self.model = Model(input = tensors, output = [pred, seq])
        self.model.compile(optimizer = 'adam', loss = {'pred_output': 'mape', 'seq_output': seq_mape}, loss_weights = {'seq_output': 3.0, 'pred_output': 1.0})
        
    def train_model_generator(self):
        '''
        To train the model, one needs to implement a generator
        The generator takes a list of tensors and outputs the list of corresponding data
        One can further pass the number of k_fold validation in the generator and split the training data and validation data correspondingly.
        '''
        generator = Generator(k_fold_number = sys.argv[1])

        train_gen = generator.gen_train(self.tensor.keys(), BATCH_SZ)
        val_gen = generator.gen_val(self.tensor.keys(), BATCH_SZ)

        nb_train, nb_val = daily_gen.get_nbs()

        checkpointer = ModelCheckpoint('./k_fold_%s.hdf5' % sys.argv[1], verbose = 1, save_best_only = True)

        self.model.fit_generator(train_gen, samples_per_epoch = nb_train, nb_epoch = NB_EPOCH,\
                validation_data = val_gen, nb_val_samples = nb_val, \
                max_q_size = 512, callbacks = [checkpointer])

def run():
    model = DeepTTE()
    model.build_model()
    model.train_model_generator()

if __name__ == '__main__':
    run()

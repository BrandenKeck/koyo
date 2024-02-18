###
# TODO TODO TODO
###

# Import model
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dropout, Dense, Embedding

# Custom Loss Callback
class LossCallback(tf.keras.callbacks.Callback):

    def __init__(self, factor=0.6):
        self.factor = factor

    def on_epoch_end(self, epoch, logs={}):
        logs['loss_factor'] = self.factor * logs['val_loss'] + (1-self.factor) * logs['loss']

# Object definition
class network_model():

    def __init__(self, config, reset=False):

        # Setup dataset
        self.sk_id = config['sk_id']
        self.mod_id = config['mod_id']
        start = config['start']
        prefactors = config['prefactors']
        postfactors = config['postfactors']
        response = config['response']
        data = config['data']
        self.set_data(start, prefactors, postfactors, response, data)

        # Init distribution parameters
        self.load_model()
        self.load_params()
        self.eval = None

        # Set Default Parameters
        if not self.params:
            self.params = {
                'dropout': 0.3,
                'l2reg': 1e-3,
                'lr': 4.12e-3
            }

        # Build New Model if Needed
        if reset or not self.model:
            self.model = self.build_model()

    def set_data(self, start, prefactors, postfactors, response, data):

        # Create X
        idx = data.index
        X = data.loc[idx[start]:idx[len(idx)-2], prefactors].to_numpy()
        for factor in postfactors:
            catcol = data.loc[idx[start+1]:, factor[0]].to_numpy().astype(int)
            cats = Embedding(factor[1], 2, input_length=1)(catcol)
            X = np.concatenate((X, cats), axis=1)
        self.X = X

        # Create Y
        y = data.loc[idx[start+1]:, response].to_numpy(dtype='int32')
        y = np.minimum(y, 4*np.ones(y.shape, dtype='int32'))
        y_cat = np.zeros((y.shape[0], 5))
        for i, j in enumerate(y): 
            if j==0: 
                y_cat[i, 0] = 1
            else:
                for k in np.arange(1, j+1):
                    y_cat[i, k]=k/np.math.factorial(j)
        self.Y = y_cat

    def build_model(self):
        
        # Input level
        x_in = Input(shape=(self.X.shape[1],))

        # Model Y
        y_dense1 = Dense(units=64, activation="relu", kernel_regularizer=regularizers.L2(self.params['l2reg']))(x_in)
        y_drop1 = Dropout(self.params['dropout'])(y_dense1)
        y_dense2 = Dense(units=32, activation="relu", kernel_regularizer=regularizers.L2(self.params['l2reg']))(y_drop1)
        y_drop2 = Dropout(self.params['dropout'])(y_dense2)
        y_dense3 = Dense(units=16, activation="relu", kernel_regularizer=regularizers.L2(self.params['l2reg']))(y_drop2)
        y_dense3 = Dense(units=8, activation="relu", kernel_regularizer=regularizers.L2(self.params['l2reg']))(y_drop2)
        y_out = Dense(units=self.Y.shape[1], activation="softmax")(y_dense3)

        # Generate the final models
        model = tf.keras.Model(inputs = x_in, outputs = y_out)

        # Model Training Params
        optim = keras.optimizers.Adam(learning_rate=self.params['lr'])
        loss_function = tf.losses.CategoricalCrossentropy()
        model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['accuracy'])
        return model

    def train(self, epochs=10000, patience=3000, validate=False, vsplit=0.2):

        # Housekeeping
        warnings.filterwarnings("ignore")
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=vsplit, shuffle=True)

        # Validation trigger (splits or no splits)
        if validate:
            mons = 'loss_factor'
            x = x_train
            y = y_train
            vdat = (x_test, y_test)
        else:
            mons = 'loss'
            x = self.X
            y = self.Y
            vdat = None
            vsplit = 0.0

        # Define callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor=mons, patience=patience)
        mc = tf.keras.callbacks.ModelCheckpoint(f'models/{self.sk_id}/{self.mod_id}/model.h5', monitor=mons, save_best_only=True)
        if validate:
            lcb = LossCallback()
            callbacks = [lcb, es, mc]
        else: callbacks = [es, mc]

        # Train network
        self.model.fit(
            x=x,
            y=y,
            verbose=0,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=vdat,
            validation_split=vsplit,
            steps_per_epoch=None,
            use_multiprocessing = True,
            batch_size=np.min([32, x.shape[0]])
        )
        self.model = load_model(f'models/{self.sk_id}/{self.mod_id}/model.h5')
        self.evaluate()

    def evaluate(self):
        self.eval = self.model.evaluate(self.X, self.Y, verbose=0)

    def predict(self, X):
        if X.ndim > 1: X = np.array(X).reshape((X.shape[0], X.shape[1],))
        else: X = np.array(X).reshape((1, X.shape[0],))
        dist = self.model.predict(X, verbose=0)
        return dist

    def load_model(self):
        try: self.model = load_model(f'models/{self.sk_id}/{self.mod_id}/model.h5')
        except: self.model = None

    def load_params(self):
        try:
            f = open(f'models/{self.sk_id}/{self.mod_id}/params.json')
            self.params = json.load(f)
        except:
            self.params = None

    def save_model(self):
        self.model.save(f'models/{self.sk_id}/{self.mod_id}/model.h5')

    def save_params(self):
        with open(f'models/{self.sk_id}/{self.mod_id}/params.json', 'w') as fp:
            json.dump(self.params, fp)
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:20:27 2018

@author: Administrator
"""

import pickle
import numpy as np
import keras
from keras.layers import Input, Dense, Masking, concatenate
from keras.models import Model



a = 56 
lstm_size = 24
max_total_char_count = 40


###############################################################################
## Das neuronale Netz
###############################################################################
def prepare_model():
    """prepares the model, has three inputs and one output in this form
    I removed the bigger inputs for simplicity
    """
    ins = []
    # Zeugs -------------------------------------------------------------------
    # Zusammenbauen
    in_einzel = Input(shape=(1,))
    in_anzahl = Input(shape=(1,))
    out_einzel = Dense(1,activation='sigmoid')(Masking()(in_einzel))
    out_anzahl = Dense(1,activation='sigmoid')(Masking()(in_anzahl))
    
    ins.append(in_einzel)
    ins.append(in_anzahl)
    # Gesamtbetrag ------------------------------------------------------------
    in_betrag = Input(shape=(1,))
    out_betrag = Dense(1,activation='sigmoid')(Masking()(in_betrag)) # um affine transformationen zu ermöglichen
    
    ins.append(in_betrag)
    
    print('Sonstiger Kack Done')
    # Kacke konkatenieren + Output --------------------------------------------
    gesamt_out = concatenate([out_einzel, out_anzahl, out_betrag])
    
    # Output ------------------------------------------------------------------
    klassen = 5
    out_pos=Dense(klassen,activation='softmax')(gesamt_out)
    
    print('Out Done, starting compiling...')
    # jetzt das Modell bauen --------------------------------------------------
    model = Model(inputs=ins, outputs=out_pos)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[keras.metrics.categorical_accuracy])
    # load weights (for demonstration)
    model.load_weights('weights.h5')
    print('Weights loaded')
    
    return model

###############################################################################
## Data preparation
###############################################################################
def get_data():
    """just loads the data from file, 10 samples in total, for demonstration
    purposes"""
    x = pickle.load(open('data_x.pickle','rb'))
    y = pickle.load(open('data_y.pickle','rb'))
    return x,y

###############################################################################
## Analyse
###############################################################################
def comp_acc(pred,true):
    """since keras evaluate is bugged, computes the accuracy by hand
    assumes equal shapes of both things"""
    return sum(np.argmax(pred, axis=1)==np.argmax(true,axis=1))/len(pred)


###############################################################################


x,y = get_data()
print('Training Data prepared')
model = prepare_model()
print('Modle prepared')

print('Total:')
print(model.evaluate(x,y, verbose=0))
print(comp_acc(model.predict(x),y))
print('Individual:')
for i in range(0,10):
    tmp_x_klein = [x[0][i:i+1],x[1][i:i+1],x[2][i:i+1]]
    pred_klein = model.predict(tmp_x_klein)
    true_y_klein = y[i:i+1]
    print(model.evaluate(tmp_x_klein,true_y_klein, verbose=0))
    print(comp_acc(model.predict(tmp_x_klein),true_y_klein))
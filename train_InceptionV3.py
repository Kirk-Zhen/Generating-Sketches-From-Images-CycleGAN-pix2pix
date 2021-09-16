import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from numpy.random import randint
from LoadData import LoadData
import math


def get_accr(images, labels, path, n_split=2):
    model = load_model("models/InceptionV3.h5")
    cust = {'InstanceNormalization': InstanceNormalization}
    A_to_B = load_model(path, cust)

    n_part = math.floor(images.shape[0] / n_split)
    for i in range(n_part):
        ix_start, ix_end = i * n_split, i * n_split + n_split
        X_in = (images[ix_start:ix_end] - 127.5) / 127.5
        if "con_CycleGAN" in path:
            X_out = A_to_B.predict([X_in, labels[ix_start:ix_end]])
        else:
            X_out = A_to_B.predict(X_in)
        X_out = (X_out + 1) / 2.0
        pred = model.predict(X_out)
        if i == 0:
            yhat = pred
        else:
            yhat = np.concatenate((yhat, pred), 0)

    total = yhat.shape[0]
    answer = np.argmax(yhat, axis=1)
    correct = np.sum(answer == labels[0: yhat.shape[0]])
    accr = correct / total
    return accr


# load data
load_data = LoadData()
train_images, _, train_labels = load_data.getData(mode='train')

b = np.zeros((train_labels.shape[0], 125))
b[np.arange(train_labels.size), train_labels] = 1
train_labels = b

img_height, img_width = train_images.shape[1], train_images.shape[2]

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.6)(x)  # latest parameter update
predictions = Dense(125, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
plot_model(model, show_shapes=True, to_file='models_plot/InceptionV3.png')
model.summary()

for ep in range(100):
    n_part = math.ceil(train_images.shape[0] / 1280)
    for i in range(n_part):
        ix1 = randint(0, train_images.shape[0], 1280)
        tr_I = train_images[ix1] / 255
        tr_L = train_labels[ix1]
        model.fit(tr_I, tr_L, epochs=1, batch_size=32)  # batch size

# Saving the model
model.save('models/InceptionV3.h5')

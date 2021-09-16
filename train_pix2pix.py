from LoadData import LoadData
import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, \
    LeakyReLU
from matplotlib import pyplot
from keras.utils import plot_model
import math


def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([in_src_image, in_target_image])

    # See report 4.1.2 the structure of discriminator:
    # C64(without batch-normalization)−C128 − C256 − C512 − C512

    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.summary()
    plot_model(model, show_shapes=True, to_file='models_plot/dis_pix2pix.png')
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# see Figure.5 or applendix for the architecture for a encoder block in U-Net
def encoder_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


# see  Figure.5 or applendix for the architecture for a decoder block in U-Net
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


def generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    # encoder part, refer to report 4.1.1
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128, batchnorm=True)
    e3 = encoder_block(e2, 256, batchnorm=True)
    e4 = encoder_block(e3, 512, batchnorm=True)
    e5 = encoder_block(e4, 512, batchnorm=True)
    e6 = encoder_block(e5, 512, batchnorm=True)
    e7 = encoder_block(e6, 512, batchnorm=True)

    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    # decoder part, refer to report 4.1.1
    d1 = decoder_block(b, e7, 512, dropout=True)
    d2 = decoder_block(d1, e6, 512, dropout=True)
    d3 = decoder_block(d2, e5, 512, dropout=True)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # d1 = decoder_block(b, e7, 1024, dropout=True)
    # d2 = decoder_block(d1, e6, 1024, dropout=True)
    # d3 = decoder_block(d2, e5, 1024, dropout=True)
    # d4 = decoder_block(d3, e4, 1024, dropout=False)
    # d5 = decoder_block(d4, e3, 512, dropout=False)
    # d6 = decoder_block(d5, e2, 256, dropout=False)
    # d7 = decoder_block(d6, e1, 128, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    model.summary()
    plot_model(model, show_shapes=True, to_file='models_plot/gen_pix2pix.png')
    return model


# connect the Generator and the Discriminator
def pix2pix(pix2pix_gen, pix2pix_dis, image_shape):
    for layer in pix2pix_dis.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = pix2pix_gen(in_src)
    dis_out = pix2pix_dis([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.summary()
    # plot_model(model, show_shapes=True, to_file='models_plot/pix2pix.png')
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


# get images and real sketches
def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = (trainA[ix] - 127.5) / 127.5, (trainB[ix] - 127.5) / 127.5
    # generate class labels for "True"
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# get images and generated sketches
def generate_fake_samples(pix2pix_gen, samples, patch_shape):
    # generate fake sketches
    X = pix2pix_gen.predict(samples)
    # generate class labels for "False"
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# generate samples and save as a plot and save the model
def plot_process(epoch, pix2pix_gen, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(pix2pix_gen, X_realA, 1)
    X_realA = (X_realA + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(X_fakeB[i])
    fn = 'process/pix2pix_plot_{}.png'.format(epoch + 1)
    pyplot.savefig(fn)
    pyplot.show()
    pyplot.close()


def save_model(epoch, pix2pix_gen):
    # save the generator model
    fn = 'models/pix2pix_Generator_{}.h5'.format(epoch + 1)
    pix2pix_gen.save(fn)


# train pix2pix
def train(pix2pix_dis, pix2pix_gen, pix2pix_model, dataset, n_epochs=5, n_batch=5):
    record = open('loss_records/loss_pix2pix.csv', 'w')
    record.write('iter,d1,d2,g\n')
    record.flush()
    curr_epoch = 0
    n_patch = pix2pix_dis.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(pix2pix_gen, X_realA, n_patch)
        d_loss1 = pix2pix_dis.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = pix2pix_dis.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = pix2pix_model.train_on_batch(X_realA, [y_real, X_realB])
        print('Iter-{}: DisA_Loss:[{:.3f}, {:.3f}], Gen_Loss:[{:.3f}]'.format(i + 1, d_loss1, d_loss2, g_loss))
        record.write("{},{},{}\n".format(i + 1, d_loss1, d_loss2, g_loss))
        record.flush()
        if (i + 1) % (bat_per_epo) == 0:
            plot_process(curr_epoch, pix2pix_gen, dataset)
            save_model(curr_epoch, pix2pix_gen)
            curr_epoch += 1
    record.close()


# load data
load_data = LoadData()
train_images, train_sketches, train_labels = load_data.getData(mode='train')
test_images, _, _ = load_data.getData(mode='test')

image_shape = train_images.shape[1:]
sket_shape = train_sketches.shape[1:]
pix2pix_dis = discriminator(sket_shape)
pix2pix_gen = generator(image_shape)
pix2pix_model = pix2pix(pix2pix_gen, pix2pix_dis, image_shape)

train(pix2pix_dis, pix2pix_gen, pix2pix_model, [train_images, train_sketches], n_batch=5, n_epochs=5)

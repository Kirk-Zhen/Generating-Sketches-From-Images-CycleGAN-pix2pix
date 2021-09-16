from LoadData import LoadData
import numpy as np
from random import random
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from keras.utils import plot_model
from shutil import copyfile
import math
from train_InceptionV3 import get_accr


def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    # See report 4.2.2 the structure of discriminator:
    # C64(without instance-normalization)−C128 − C256 − C512 − C512

    # C64(without instance-normalization)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # patch_out = Activation('sigmoid')(patch_out)
    model = Model(in_image, patch_out)
    model.summary()
    # plot the overall structure
    plot_model(model, show_shapes=True, to_file='models_plot/dis_CycleGAN.png')
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


#  Figure 9 in the report
def residual_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # skip connection
    g = Concatenate()([g, input_layer])
    return g


# define the standalone generator model
def generator(image_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c3s2-128
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c3s2-256
    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = residual_block(256, g)
    # tc3s2-128
    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # tc3s2-64
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)

    # plot the overall structure
    model.summary()
    plot_model(model, show_shapes=True, to_file='models_plot/gen_CycleGAN.png')
    # model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def oneway_CycleGAN(generator_1, d_model, generator_2, image_shape):
    generator_1.trainable = True
    d_model.trainable = False
    generator_2.trainable = False
    input_gen = Input(shape=image_shape)
    gen1_out = generator_1(input_gen)
    output_d = d_model(gen1_out)
    input_id = Input(shape=image_shape)
    output_id = generator_1(input_id)
    # forward cycle
    output_f = generator_2(gen1_out)
    # backward cycle
    gen2_out = generator_2(input_id)
    output_b = generator_1(gen2_out)
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.summary()
    # plot_model(model, show_shapes=True, to_file='models_plot/CycleGAN.png')
    # weight of the loss function, addjust here to modify the importance weight we mentioned in 7.2
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model


# For investigating the performance druing training proces
def sample_for_test(dataset):
    X = (dataset[[12944, 19338, 4008, 15985, 1644, 6702, 9704]] - 127.5) / 127.5
    return X


# get images and real sketches
def generate_real_samples(dataset, num_img, patch_shape):
    # choose random instances
    idx = randint(0, dataset.shape[0], num_img)
    X = (dataset[idx] - 127.5) / 127.5
    # generate class labels for "True"
    y = np.ones((num_img, patch_shape, patch_shape, 1))
    return X, y


# generate a batch of images, returns images and targets
def generate_fake_samples(generator, dataset, patch_shape):
    # generate fake sketches
    X = generator.predict(dataset)
    # generate class labels for "False"
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# save the generator models to file
def save_models(epoch, generator):
    # save the first generator model
    filename1 = 'models/CycleGAN_Sketch_Generator_{}.h5'.format(epoch + 1)
    generator.save(filename1)


def plot_process(epoch, generator, trainX, name, num_img=7):
    img_in = sample_for_test(trainX)
    # generate sketches
    skt_out, _ = generate_fake_samples(generator, img_in, 0)
    # scale from [-1,1] to [0,1]
    img_in = (img_in + 1) / 2.0
    skt_out = (skt_out + 1) / 2.0
    # plot real images
    for i in range(num_img):
        pyplot.subplots_adjust(wspace=0, hspace=0)
        pyplot.subplot(num_img - 2, num_img, 1 + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(img_in[i])
    # plot translated image
    for i in range(num_img):
        pyplot.subplots_adjust(wspace=0, hspace=0)
        pyplot.subplot(num_img - 2, num_img, 1 + num_img + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(skt_out[i])
    # save plot to file
    filename1 = 'process/CycleGAN_{}_plot_{}.png'.format(name, (epoch + 1))
    # pyplot.tight_layout()
    pyplot.savefig(filename1, dpi=300)
    pyplot.show()
    pyplot.close()


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            idx = randint(0, len(pool))
            selected.append(pool[idx])
            pool[idx] = image
    return np.asarray(selected)


# def train(discriminator_A, discriminator_B, generator_AtoB, generator_BtoA, cycle_AtoB, cycle_BtoA, dataset, dic):
def train(discriminator_A, discriminator_B, generator_AtoB, generator_BtoA, cycle_AtoB, cycle_BtoA, dataset, n_epochs=5, n_batch=5):
    # record loss over training process
    record = open('loss_records/loss_CycleGAN.csv', 'w')
    record.write('iter,dA1,dA2,dB1,dB2,g1,g2\n')
    record.flush()
    n_patch = discriminator_A.output_shape[1]
    trainA, trainB = dataset
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    curr_epoch = 0
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(generator_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(generator_AtoB, X_realA, n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _ = cycle_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = discriminator_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = discriminator_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = cycle_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = discriminator_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = discriminator_B.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        print("Iter-{}: DisA_Loss:[{:.3f}, {:.3f}], DisB_Loss[{:.3f}, {:.3f}], Gen_Loss:[{:.3f}, {:.3f}]".format(i + 1,
                                                                                                                 dA_loss1,
                                                                                                                 dA_loss2,
                                                                                                                 dB_loss1,
                                                                                                                 dB_loss2,
                                                                                                                 g_loss1,
                                                                                                                 g_loss2))
        # write the csv to record the loss
        record.write("{},{},{},{},{},{},{}\n".format(i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        record.flush()
        # visalize the performance and svae the generator for each epoch
        if (i + 1) % (bat_per_epo) == 0:
            plot_process(curr_epoch, generator_AtoB, trainA, 'AtoB')
            save_models(curr_epoch, generator_AtoB)
            curr_epoch += 1
    record.close()


# load data
load_data = LoadData()
train_images, train_sketches, train_labels = load_data.getData(mode='train')
test_images, _, test_labels = load_data.getData(mode='test')

# connect the CycleGAN
image_shape = train_images.shape[1:]
sket_shape = train_sketches.shape[1:]
# two generator
generator_AtoB = generator(image_shape)
generator_BtoA = generator(image_shape)
# two discriminator
discriminator_A = discriminator(image_shape)
discriminator_B = discriminator(image_shape)

cycle_AtoB = oneway_CycleGAN(generator_AtoB, discriminator_B, generator_BtoA, image_shape)
cycle_BtoA = oneway_CycleGAN(generator_BtoA, discriminator_A, generator_AtoB, image_shape)

# train models
train(discriminator_A, discriminator_B, generator_AtoB, generator_BtoA, cycle_AtoB, cycle_BtoA, [train_images, train_sketches])
accr = get_accr(test_images, test_labels, 'models/CycleGAN_Sketch_Generator_5.h5')
print(accr)

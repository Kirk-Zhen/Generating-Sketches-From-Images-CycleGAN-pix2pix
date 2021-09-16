from LoadData import LoadData
from random import random
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D, Embedding, Dense, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Reshape
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from keras.utils import plot_model
import numpy as np
import math
from train_InceptionV3 import get_accr


def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    # See report 4.2.2 the structure of discriminator:
    # Which is exact the same as CycleGAN
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
    plot_model(model, show_shapes=True, to_file='models_plot/dis_con_CycleGAN.png')
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
def generator(image_shape, n_resnet=9, n_classes=125):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    # g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(merge)
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

    # concatenate label projection as a extra channel

    in_label = Input(shape=(1,))
    # embedding for categories
    li = Embedding(n_classes, n_classes)(in_label)
    # scale up to destinated dimension
    n_nodes = 64 * 64
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((64, 64, 1))(li)
    g = Concatenate()([g, li])

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

    model = Model([in_image, in_label], out_image)
    model.summary()
    plot_model(model, show_shapes=True, to_file='models_plot/gen_con_CycleGAN.png')
    return model


# define a composite model for updating generators by adversarial and cycle loss
def one_way_model(generator_1, discriminator, generator_2, image_shape):
    generator_1.trainable = True
    discriminator.trainable = False
    generator_2.trainable = False

    input_gen = Input(shape=image_shape), Input(shape=(1,))
    gen1_out = generator_1(input_gen)
    output_d = discriminator(gen1_out)
    input_id = Input(shape=image_shape), Input(shape=(1,))
    output_id = generator_1(input_id)
    # forward cycle
    output_f = generator_2([gen1_out, input_gen[1]])
    # backward cycle
    gen2_out = generator_2(input_id)
    output_b = generator_1([gen2_out, input_id[1]])
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.summary()
    # plot_model(model, show_shapes=True, to_file='models_plot/con_CycleGAN.png')
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model


def sample_for_test(dataset, num_images, patch_shape, label):
    X = (dataset[[12944, 19338, 4008, 15985, 1644]] - 127.5) / 127.5
    L = label[[12944, 19338, 4008, 15985, 1644]]
    y = np.ones((num_images, patch_shape, patch_shape, 1))
    return X, y, L


def generate_real_samples_mod(setA, setB, num_images, patch_shape, label):
    idx = np.random.randint(0, setA.shape[0], num_images)
    xA = (setA[idx] - 127.5) / 127.5
    xB = (setB[idx] - 127.5) / 127.5
    L = label[idx]
    yA = np.ones((num_images, patch_shape, patch_shape, 1))
    yB = np.ones((num_images, patch_shape, patch_shape, 1))
    return xA, xB, yA, yB, L, L


# generate a batch of images, returns images and targets
def generate_fake_samples(generator, dataset, patch_shape):
    X = generator.predict(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, dataset[1], y


# save the generator models to file
def save_models(epoch, generator_AtoB):
    # save the first generator model
    filename1 = 'models/con_CycleGAN_Sketch_Generator_{}.h5'.format(epoch + 1)
    generator_AtoB.save(filename1)


# generate samples and save as a plot and save the model
def plot_process(epoch, generator, trainX, labels, name, num_images=5):
    # select a sample of input images
    img_in, _, L_in = sample_for_test(trainX, num_images, 0, labels)
    # generate translated images
    skt_out, _, _ = generate_fake_samples(generator, [img_in, L_in], 0)
    # scale all pidxels from [-1,1] to [0,1]
    img_in = (img_in + 1) / 2.0
    skt_out = (skt_out + 1) / 2.0
    for i in range(num_images):
        pyplot.subplots_adjust(wspace=0, hspace=0)
        pyplot.subplot(num_images - 2, num_images, 1 + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(img_in[i])
        # plot translated image
    for i in range(num_images):
        pyplot.subplots_adjust(wspace=0, hspace=0)
        pyplot.subplot(num_images - 2, num_images, 1 + num_images + i)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.imshow(skt_out[i])

    # save plot to file
    filename1 = 'process/con_CycleGAN_{}_plot_{}.png'.format(name, (epoch + 1))
    pyplot.savefig(filename1, dpi=300)
    pyplot.show()
    pyplot.close()


# update image pool for fake images
def update_image_pool(pool, images, max_size=100):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            idx = np.random.randint(0, len(pool))
            selected.append(pool[idx])
            pool[idx] = image
    return np.asarray(selected)


# train cyclegan models
def train(discriminator_A, discriminator_B, generator_AtoB, generator_BtoA, cycle_AtoB, cycle_BtoA, dataset, n_epochs=5, n_batch=5):
    record = open('loss_records/loss_con_CycleGAN.csv', 'w')
    record.write('iter,dA2,dB2,g1,g2\n')
    record.flush()
    # define properties of the training run
    # determine the output square shape of the discriminator
    n_patch = discriminator_A.output_shape[1]
    # print(discriminator_A.output_shape)
    # unpack dataset
    trainA, trainB, labels = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    curr_epoch = 0
    for i in range(n_steps):
        X_realA, X_realB, y_realA, y_realB, L_A, L_B = generate_real_samples_mod(trainA, trainB, n_batch, n_patch,
                                                                                 labels)
        X_fakeA, L1_A, y_fakeA = generate_fake_samples(generator_BtoA, [X_realB, L_A], n_patch)
        X_fakeB, L1_B, y_fakeB = generate_fake_samples(generator_AtoB, [X_realA, L_B], n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _ = cycle_BtoA.train_on_batch([[X_realB, L_A], [X_realA, L_A]],
                                                          [y_realA, X_realA, X_realB, X_realA])
        dA_loss1 = discriminator_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = discriminator_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = cycle_AtoB.train_on_batch([[X_realA, L_B], [X_realB, L_B]],
                                                          [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = discriminator_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = discriminator_B.train_on_batch(X_fakeB, y_fakeB)
        print("Iter-{}: DisA_Loss:[{:.3f}, {:.3f}], DisB_Loss[{:.3f}, {:.3f}], Gen_Loss:[{:.3f}, {:.3f}]".format(i + 1,
                                                                                                                 dA_loss1,
                                                                                                                 dA_loss2,
                                                                                                                 dB_loss1,
                                                                                                                 dB_loss2,
                                                                                                                 g_loss1,
                                                                                                                 g_loss2))
        record.write("{},{},{},{},{}\n".format(i + 1, dA_loss2, dB_loss2, g_loss1, g_loss2))
        record.flush()
        if (i + 1) % (bat_per_epo) == 0:
            plot_process(curr_epoch, generator_AtoB, trainA, labels, 'AtoB')
            save_models(curr_epoch, generator_AtoB)
            curr_epoch += 1
    record.close()


# # load data
load_data = LoadData()
train_images, train_sketches, train_labels = load_data.getData(mode='train')
test_images, _, test_labels = load_data.getData(mode='test')

image_shape = train_images.shape[1:]
sket_shape = train_sketches.shape[1:]
generator_AtoB = generator(image_shape)
generator_BtoA = generator(image_shape)
discriminator_A = discriminator(image_shape)
discriminator_B = discriminator(image_shape)
cycle_AtoB = one_way_model(generator_AtoB, discriminator_B, generator_BtoA, image_shape)
cycle_BtoA = one_way_model(generator_BtoA, discriminator_A, generator_AtoB, image_shape)

# train models
train(discriminator_A, discriminator_B, generator_AtoB, generator_BtoA, cycle_AtoB, cycle_BtoA,
      [train_images, train_sketches, train_labels])
accr = get_accr(test_images, test_labels, 'models/con_CycleGAN_Sketch_Generator_5.h5')
print(accr)

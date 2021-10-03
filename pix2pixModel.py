
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras import *

from networks import *

import argparse

#Example command
#python3 pix2pixModel.py --save_ck 1 --new 1 --save_model "Model-Red-VGA-8-E5" --epochs 5 --n 4500


#Arguments for command line
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int,
                    default=25,
                    help='Number of epochs to run the training')

parser.add_argument('--batch_size', type=int,
                    default=6,
                    help='Size of batch to run the training')

parser.add_argument('--n', type=int,
                    default=1000,
                    help='Number of samples')

parser.add_argument('--new', type=bool,
                    default=False,
                    help='Training from Scratch or from Checkpoint')

parser.add_argument('--save_ck', type=bool,
                    default=False,
                    help='Save Checkpoint')

parser.add_argument('--save_model', type=str,
                    default="",
                    help='Save Model')

args = parser.parse_args()

#Directories declaration
PATH = "data_recon"
INPATH = r"C:\Users\fuliw\Documents\python\Data_Generation\Data\DATA_RED_8_VGA" #input images
OUTPATH = r"C:\Users\fuliw\Documents\python\Data_Generation\Data\DATA_RED_8_VGA_GT"  # ground truth

ckpt_name = "./ckpt/tf_ckpts_rot_vga"
output_name = "./output/output_rot_vga"

DPATH = "./Data/dummyAs"

#Training data set length
n = args.n
train_n = round(n*0.90)

#List content of input path
imgurls = [f for f in os.listdir(INPATH)]

#Shuffle listings
randurls = np.copy(imgurls)
np.random.shuffle(randurls)
tr_urls = randurls[:int(train_n)]
ts_urls = randurls[int(train_n):int(n)]

print(len(tr_urls), len(ts_urls))

#Dimension of input
IMG_WIDTH = 640
IMG_HEIGHT = 480


def create_dir(dire):
#Create directory if not existent
    if not os.path.exists(dire):
        os.makedirs(dire)

def resize(inimg, tgimg, height, width):
#Resize image
    inimg = tf.image.resize(inimg, [height,width])
    tgimg = tf.image.resize(tgimg, [height,width])
    return inimg, tgimg

def normalize(inimg, tgimg):
#Normalize Image
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1
    return inimg, tgimg

def load_image(filename, augment = True):
#Load image for using for the model
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + "/" + filename)),tf.float32)[...,:3]
    tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH + "/" + filename)),tf.float32)[...,:3]

    inimg, tgimg = resize(inimg,tgimg, IMG_HEIGHT, IMG_WIDTH)
    inimg, tgimg = normalize(inimg, tgimg)

    return inimg, tgimg


def load_train_image(filename):
    return load_image(filename, True)

def load_test_image(filename):
    return load_image(filename, False)


#Declaring Data Sets
train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls = 6)
train_dataset = train_dataset.batch(args.batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls = 6)
test_dataset = test_dataset.batch(args.batch_size)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discrimiator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):

    gen_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gen_loss + (LAMBDA*l1_loss)

    return total_gen_loss


def generate_images(model, test_input, tar,filename, save_filename = True, display_imgs = True):
    prediction = model(test_input, training=True)

    plt.figure(figsize=(10,10))
    if save_filename:

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        create_dir(output_name)
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')


        print(output_name + "/" + filename + ".png")
        plt.savefig(output_name + "/" + filename + ".png")



    if display_imgs:
        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])

            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


from IPython.display import clear_output

generator = GeneratorHD()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

checkpoint = tf.train.Checkpoint(
                                step = tf.Variable(1),
                                generator_optimizer = generator_optimizer,
                                discriminator_optimizer = discriminator_optimizer,
                                generator = generator,
                                discriminator = discriminator)


create_dir(ckpt_name)
manager = tf.train.CheckpointManager(checkpoint, ckpt_name, max_to_keep = 5)




@tf.function()
def train_step(input_image, target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:


        output_image = generator(input_image, training = True)

        output_gen_discr = discriminator([output_image, input_image], training = True)

        output_trg_discr = discriminator([target, input_image], training = True)

        discr_loss = discrimiator_loss(output_trg_discr, output_gen_discr)

        gen_loss = generator_loss(output_gen_discr, output_image, target)

        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

    return discr_loss, gen_loss


def train(dataset, epochs):

    if not args.new:
        checkpoint.restore(manager.latest_checkpoint)
        print("Restore from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from Scratch")

    for epoch in range(epochs):

        imgi = 0
        for input_image, target in dataset:

            d_loss, g_loss = train_step(input_image, target)
            print('epoch ' + str(epoch) + "/" +str(epochs) + ' - train: ' + str(imgi)+ '/' + str(len(tr_urls)/args.batch_size),d_loss, g_loss)
            imgi += 1
            clear_output(wait = True)

        imgi = 0
        for inp, tar in test_dataset.take(5):
            generate_images(generator, inp, tar,  str(imgi) + '_' + str(int(checkpoint.step)) ,save_filename = True, display_imgs = False)
            imgi += 1

        checkpoint.step.assign_add(1)

        if int(checkpoint.step) % 1 == 0 and args.save_ck:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            print("Saved checkpoint for step ")

def save_model():
    create_dir("./Models/" + args.save_model)
    tf.saved_model.save(generator, "./Models/" + args.save_model)
    print("Model Saved")


train(train_dataset, args.epochs)

if args.save_model == "":
    save_model()

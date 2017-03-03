# 500px ML Challenge by Andrey Bogomazov

# What I have learned while conpleting the challenge:
# 1. Some the front end of TensorFlow.
# 2. How Tensorflow works (computational graphs, sessions, and some other stuff that is happening in the background)
# 3. Taking derivatives is very cheap from colah's blog. (regarding backpropagation in neural nets and computational graphs)
# 4. Reviewed softmax classification.

# Notes:
# Neural network was not used as it is time consuming to train on my laptop.
# Results of the training could be saved to avoid retraining every time the code is being run.

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mimage
from matplotlib.backends.backend_pdf import PdfPages
import logging

# image dimensions
SAMPLE_DIM = 28

# Tensor Flow Tutorial code
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, SAMPLE_DIM*SAMPLE_DIM])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([SAMPLE_DIM*SAMPLE_DIM,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

# With lower regularization it is easier to trick the model
# as mentioned in the artilcle, hence was not introduced.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

SAMPLES_TWO = []

VAL_IMAGE = 0
VAL_Y = 1

# 1. Feed our model
# 2. Collect samples of 2
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if len(SAMPLES_TWO) < 10:
        for j in xrange(len(batch_xs)):
            sample = (batch_xs[j], batch_ys[j])

            if sample[VAL_Y][2] and len(SAMPLES_TWO) < 10:
                SAMPLES_TWO += [sample]
    
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

def reshape_image(image_1d):
    return tf.reshape(image_1d, [SAMPLE_DIM, SAMPLE_DIM])

def show_image(image_1d):
    image = reshape_image(image_1d)
    plt.imshow(image.eval())

    plt.savefig("fig.png")
    plt.show()

def get_adv_image(sample, delta):
    return (sample, delta, sample + delta)

def classify_image(image):
    feed_dict = {x: [image]}
    return tf.argmax(sess.run(y, feed_dict), 1).eval()

def plot_adv_images(samples):
    rows = len(samples)
    cols = len(samples[0])
    gs = gridspec.GridSpec(rows, cols, top=1., bottom=0., right=1., left=0., hspace=0.,
                wspace=0.)

    for i, g in enumerate(gs):
        ax = plt.subplot(g)
        image = reshape_image(samples[i/cols][i%cols]).eval()
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
            
    plt.savefig("adv_result.png")
    plt.show()

# Using weights of "six" class as our delta 
w_vals = sess.run(W)
w_six = [row[6] for row in w_vals]
adv_images = [get_adv_image(sample[VAL_IMAGE], w_six) for sample in SAMPLES_TWO]

# Log classification results 
# (Some adversarial images might not "trick" the model)
classified = [classify_image(adv_image[2]) for adv_image in adv_images]
logging.info('classified as {}'.format(classified))
plot_adv_images(adv_images)


# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




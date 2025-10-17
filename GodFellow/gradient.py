import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top = 1)[0][0]

image_path = tf.keras.utils.get_file('grace_hopper.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image, channels = 3)
image = preprocess(image)
image = tf.expand_dims(image, 0)
image = tf.Variable(image)

plt.figure()
plt.imshow(image[0])
plt.title(get_imagenet_label(pretrained_model.predict(image))[2])
plt.grid(False)
plt.show()
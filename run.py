import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import argparse
import numpy as np

(_,_),(x_test,y_test) = fashion_mnist.load_data()

labels = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
          5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"} 

model = tf.keras.models.load_model('src/models/fashion_mnist.h5')
x_test  = x_test/255 

def make_prediction(idx):
    pred = model.predict(x_test[idx][np.newaxis, ...],verbose=0)
    return np.argmax(pred)

def plot_img(idx):
    prediction = make_prediction(idx)
    predicted_label = labels[prediction]
    correct_label = labels[y_test[idx]]
    if predicted_label==correct_label:
        print("{:*^80}".format('Correct Prediction'))
    else:
        print("{:x^80}".format("Wrong Prediction"))
    print("predicted label : {}".format(predicted_label))
    plt.imshow(x_test[idx])
    print("original label : {}".format(correct_label))
    plt.axis('off')
    plt.show()

parser = argparse.ArgumentParser("plotting result")
parser.add_argument('--idx', type=int)

args = parser.parse_args()

plot_img(args.idx)
plt.show()
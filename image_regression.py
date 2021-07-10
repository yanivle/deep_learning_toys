import random
import numpy as np
from optimizers import AdaGrad, GradientDescent, Momentum, RMSProp, PlotLogger, default_logging_function
from models import Sequential
from layers import FullyConnected, RegressionID, Relu, Sigmoid
from PIL import Image
import tqdm
from matplotlib import pyplot as plt

model = Sequential()
model.addLayer(FullyConnected(2, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=0.5))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
model.addLayer(Relu(20))

# model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
# model.addLayer(Relu(20))
# model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
# model.addLayer(Relu(20))
# model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
# model.addLayer(Relu(20))
# model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
# model.addLayer(Relu(20))
# model.addLayer(FullyConnected(20, 20, bias_pref=0.1, epsilon_init=1))
# model.addLayer(Relu(20))

model.addLayer(FullyConnected(20, 3, bias_pref=0.1, epsilon_init=1))
model.addLayer(RegressionID(3))

NEW_WIDTH = 200
image = Image.open('pencils.png').convert('RGB')
w, h = image.size
NEW_HEIGHT = h * NEW_WIDTH // w

def xy_to_model(x, y):
    return np.array([[(x - NEW_WIDTH / 2) / NEW_WIDTH, (y - NEW_HEIGHT / 2) / NEW_HEIGHT]], dtype=np.float_).T

def rgb_to_model(r, g, b):
    return np.array([[r, g, b]], dtype=np.float_).T / 255

def model_to_rgb(y):
    prediction = (y * 255).astype(np.int64)
    np.clip(prediction, 0, 255, out=prediction)
    return prediction.T.tolist()[0]

image = image.resize((NEW_WIDTH, NEW_HEIGHT))
training_set = []
for y in tqdm.tqdm(range(image.height)):
    for x in range(image.width):
        r, g, b = image.getpixel((x, y))
        input = xy_to_model(x, y)
        target = rgb_to_model(r, g, b)
        training_set.append((input, target))

print('Total', len(training_set), 'examples')


def show_image(*args):
    default_logging_function(*args)
    print('Learning rate:', optimizer.learning_rate)
    new_image = Image.new('RGB', (NEW_WIDTH, NEW_HEIGHT))
    for y in range(new_image.height):
        for x in range(new_image.width):
            r, g, b = model_to_rgb(model.predict(xy_to_model(x, y)))
            new_image.putpixel((x, y), (r, g, b))
    ax2.imshow(new_image)
    plt.pause(0.05)


plt.show(block=False)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.axis('off')
ax2.axis('off')
ax1.imshow(image)
optimizer = Momentum(model, learning_rate=1e-3, l2_decay=0.)
optimizer.train(training_set, log_every=1_000, logging_function=show_image, epochs=30, learning_rate_decay=0.999, mini_batch_size=500)


# np.random.seed(1234)
# random.seed(1234)

# model = Sequential()
# model.addLayer(FullyConnected(1, 1))
# # model.addLayer(Relu(1))
# model.addLayer(RegressionID(1))

# training_set = []
# for i in range(10000):
#     x = np.random.uniform(-100, 100, (1, 1))
#     y = x * 30 + 5
#     training_set.append((x, y))


# # for i in range(5):
# #     print(model.details())

# #     x, y = random.choice(training_set)
# #     print('Training example:', x, y)
# #     print('Model prediction:', model.predict(x))

# #     activations, caches = model.forward(x)
# #     dparams_for_all_layers = model.backward(activations, caches, y)
# #     dparams = dparams_for_all_layers[0]
# #     print(dparams)

# #     optimizer = GradientDescent(model, learning_rate=1e-5)
# #     optimizer.update(dparams)

# print('Total', len(training_set), 'examples')
# GradientDescent(model, learning_rate=1e-5, l2_decay=0).train(training_set, print_every=10000)

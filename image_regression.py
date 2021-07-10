import random
import numpy as np
from optimizers import AdaGrad, GradientDescent, RMSProp, PlotLogger
from models import Sequential
from layers import FullyConnected, RegressionID, Relu, Sigmoid
from PIL import Image
import tqdm

model = Sequential()
model.addLayer(FullyConnected(2, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 20))
model.addLayer(Relu(20))
model.addLayer(FullyConnected(20, 3))
model.addLayer(RegressionID(3))

NEW_WIDTH = 100
image = Image.open('me.jpg')
w, h = image.size
image = image.resize((NEW_WIDTH, h * NEW_WIDTH // w))
training_set = []
for y in tqdm.tqdm(range(image.height)):
    for x in range(image.width):
        r, g, b = image.getpixel((x, y))
        input = np.array([[x, y]], dtype=np.float_).T
        target = np.array([[r, g, b]], dtype=np.float_).T
        training_set.append((input, target))

print('Total', len(training_set), 'examples')

GradientDescent(model, learning_rate=1e-5, l2_decay=0).train(training_set, log_every=1000, logging_function=PlotLogger())


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

import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from vggmodel import predictor_model

model = predictor_model()
image_gen = ImageDataGenerator()
image_gen.flow_from_directory('data/')

batch_size = 16

train_image_gen = image_gen.flow_from_directory('data/',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=150,
                              validation_data=train_image_gen,
                              validation_steps=12)

#results.history['acc']
#plt.plot(results.history['acc'])



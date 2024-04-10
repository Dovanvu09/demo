
from keras import layers
from keras import models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential



train_datagen = ImageDataGenerator(rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator( rescale = 1.0/255)
train_generator = train_datagen.flow_from_directory("C:\\Users\\Admin\\Downloads\\Dataset\\Train",
                                                    batch_size =256 ,
                                                    class_mode = 'binary', 
                                                    target_size = (64, 64))     
validation_generator =  test_datagen.flow_from_directory( "C:\\Users\\Admin\\Downloads\\Dataset\\Validation",
                                                          batch_size  = 256,
                                                          class_mode  = 'binary', 
                                                          target_size = (64, 64))
model = models.Sequential([
    layers.Conv2D(32, (3,3), input_shape = (64,64,3) ,activation = "relu"),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.1),

    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(256, (3,3), activation = 'relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Flatten(),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(2, activation  ='sigmoid')
])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
   )
hist = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=256,
                    validation_steps=256,
                    epochs=10)
# model.save("gender_classification.h5")

import matplotlib.pyplot as plt
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()
# plt.show()
# model = models.load_model("gender_classification.h5")
# from keras.preprocessing import image
# import numpy as np 
# # predicting images
# def Classifier(path):
#     img = image.load_img(path, target_size=(64, 64))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=1)
#     print(classes[0])
#     if classes[0][0]==0:
#         return True
#     else: 
#         return False

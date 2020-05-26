#!/usr/bin/env python
# coding: utf-8

# ## Seeing the World: Model Training

# ### Specify train and validate input folders

# In[1]:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str)
args = parser.parse_args()

train_input_folder = args.data_dir + 'train/'
validate_input_folder = args.data_dir + 'validate/'

##train input folder
#train_input_folder = '/data/data4/farmer_market'
#
##validation input folder
#validate_input_folder = '/data/data4/validate/farmer_market'
#
#
## In[2]:
#
#
from imutils import paths
import os
import shutil
import random

def split_data(directory, validate_directory='validation', split=0.8):
  directories = [os.path.join(directory, o) for o in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory,o))]
  for directory in directories:
    image_paths = list(paths.list_images(directory))

    random.seed(32)
    random.shuffle(image_paths)
    image_paths

    # compute the training and testing split
    i = int(len(image_paths) * split)
    train_paths = image_paths[:i]
    selected_for_validation_paths = image_paths[i:]
    for path in selected_for_validation_paths:
       category = os.path.basename(os.path.normpath(directory))
       dest_path = os.path.join(validate_directory, category)
       if not os.path.exists(dest_path):
           os.makedirs(dest_path)
           os.chmod(dest_path, 0o777)
       try:
           shutil.move(path, dest_path)
       except OSError as e:
           if e.errno == errno.EEXIST:
               print('Image already exists.')
           else:
               raise




# In[3]:


#split_data(directory=train_input_folder,
#           validate_directory= validate_input_folder)


# ### Create train and validate data generators

# In[4]:


#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
##apply image augmentation
#train_image_generator = ImageDataGenerator(
#    rescale=1./255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    brightness_range=[0.5, 1.5],
#    horizontal_flip=True,
#    vertical_flip=True,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2)
#
#validate_image_generator = ImageDataGenerator(rescale=1./255)
#
#
#
## In[5]:
#
#
#batch_size = 5#30
#image_width = 224
#image_height = 224
#IMAGE_WIDTH_HEIGHT = (image_width, image_height)
#
#class_mode = 'categorical'
#
##create train data generator flowing from train_input_folder
#train_generator = train_image_generator.flow_from_directory(
#            train_input_folder,
#            target_size=IMAGE_WIDTH_HEIGHT,
#            batch_size=batch_size,
#            class_mode=class_mode)
#
##create validation data generator flowing from validate_input_folder
#validation_generator = validate_image_generator.flow_from_directory(
#        validate_input_folder,
#        target_size=IMAGE_WIDTH_HEIGHT,
#        batch_size=batch_size,
#        class_mode=class_mode)
#
#
## ### Create Custom Model
#
## In[6]:
#
#
#from tensorflow.keras import layers
#from tensorflow.keras import Model
#from tensorflow.keras.optimizers import Adam
#
#total_classes = 60
#activation_function = 'softmax'
#loss = 'categorical_crossentropy'
#
#img_input = layers.Input(shape=(image_width, image_height, 3))
#
#x = layers.Conv2D(32, 3, activation='relu')(img_input)
#x = layers.MaxPooling2D(2)(x)
#
#x = layers.Conv2D(64, 3, activation='relu')(x)
#x = layers.MaxPooling2D(2)(x)
#
#x = layers.Flatten()(x)
#
#x = layers.Dense(512, activation='relu')(x)
#
#x = layers.Dropout(0.5)(x)
#
#output = layers.Dense(total_classes, activation=activation_function)(x)
#
#model = Model(img_input, output)
#model.compile(loss=loss,
#              optimizer=Adam(lr=0.001),
#              metrics=['accuracy'])
#
#
## ### Train Custom Model
#
## In[8]:
#
#
#import os, datetime
#import tensorflow as tf
#
#epochs = 5
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
#
#logdir = os.path.join("tf_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#
#print('Started Training')
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[tensorboard_callback],
#      epochs=epochs)
#
#
## In[ ]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 1.0])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')


# ### Using Transfer Learning

# In[6]:


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19

image_width=224
image_height=224
IMAGE_SHAPE = (image_width, image_height, 3)

base_model = tf.keras.applications.VGG19(input_shape=IMAGE_SHAPE, include_top=False,weights='imagenet')

base_model.summary()


# In[7]:


keras = tf.keras
IMAGE_WIDTH_HEIGHT = (image_width, image_height)
batch_size=30
class_mode="categorical"

total_classes = 64
activation_function = 'softmax'
loss = 'categorical_crossentropy'


train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg19.preprocess_input,
            rescale=1.0/255.0,
            shear_range=0.2,
            zoom_range=[0.9, 1.25],
            brightness_range=[0.5, 1.5],
            horizontal_flip=True,
            vertical_flip=True)


validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg19.preprocess_input,
            rescale=1.0/255.0)


train_generator = train_image_generator.flow_from_directory(
            train_input_folder,
            target_size=IMAGE_WIDTH_HEIGHT,
            batch_size=batch_size,
            class_mode=class_mode)
validation_generator = validation_image_generator.flow_from_directory(
            validate_input_folder,
            target_size=IMAGE_WIDTH_HEIGHT,
            batch_size=batch_size,
            class_mode=class_mode)


# In[8]:


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import os

reload_checkpoint=True

total_classes=64
img_input = layers.Input(shape=(image_width, image_height, 3))

global_average_layer = layers.GlobalAveragePooling2D()
prediction_layer = layers.Dense(total_classes, activation='softmax')


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

checkpoint_path = args.data_dir+"train_model_fruit_veggie_9/chkpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


if (reload_checkpoint and os.path.isdir(checkpoint_path)):
   try:
      model.load_weights(checkpoint_path)
      print('loaded weights from checkpoint')
   except Exception:
      print('no checkpointed weights')
      pass

if not os.path.isdir(checkpoint_path):
   os.makedirs(checkpoint_path)


print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = False

model.compile(loss=loss,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()






# In[9]:


import datetime, os

epochs = 80
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size
#steps_per_epoch = 5
#validation_steps = 5

logdir = os.path.join(args.data_dir+"tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, save_best_only=True,
                                                         verbose=1)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      validation_data=validation_generator,
      validation_steps=validation_steps,
      callbacks=[checkpoint_callback, tensorboard_callback],
      epochs=epochs)


# In[10]:


#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb; pdb.set_trace()
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 3])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('data.png')

# ### Continue Training

# In[11]:


#import datetime, os
#
#epochs = 20
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
##steps_per_epoch = 50
##validation_steps = 50
#
#logdir = os.path.join("/data/tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                         save_weights_only=True, save_best_only=True,
#                                                         verbose=1)
#
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[checkpoint_callback, tensorboard_callback],
#      epochs=epochs)
#
#
#
## In[12]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 3])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#
#
## ### Continue Training
#
## In[13]:
#
#
#import datetime, os
#
#epochs = 20
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
##steps_per_epoch = 50
##validation_steps = 50
#
#logdir = os.path.join("/data/tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                         save_weights_only=True, save_best_only=True,
#                                                         verbose=1)
#
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[checkpoint_callback, tensorboard_callback],
#      epochs=epochs)
#
#
#
## In[14]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 3])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#
#
## ### Continue Training
#
## In[15]:
#
#
#import datetime, os
#
#epochs = 20
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
##steps_per_epoch = 50
##validation_steps = 50
#
#logdir = os.path.join("/data/tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                         save_weights_only=True, save_best_only=True,
#                                                         verbose=1)
#
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[checkpoint_callback, tensorboard_callback],
#      epochs=epochs)
#
#
#
## In[16]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 3])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#
#
## ### Continue Training
#
## In[17]:
#
#
#import datetime, os
#
#epochs = 20
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
##steps_per_epoch = 50
##validation_steps = 50
#
#logdir = os.path.join("/data/tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                         save_weights_only=True, save_best_only=True,
#                                                         verbose=1)
#
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[checkpoint_callback, tensorboard_callback],
#      epochs=epochs)
#
#
#
## In[18]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 3])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#
#
## ### Fine Tuning
#
## In[19]:
#
#
#import datetime, os
#
#loss = 'categorical_crossentropy'
#
#checkpoint_path = "/data/train_model_fruit_veggie_9/chkpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#
#
#if (reload_checkpoint and os.path.isdir(checkpoint_path)):
#   try:
#      model.load_weights(checkpoint_path)
#   except Exception:
#      pass
#
#if not os.path.isdir(checkpoint_path):
#   os.makedirs(checkpoint_path)
#
#base_model.trainable = True
#
## Fine tune start from layer 10
#fine_tune_at = 10
#
## Freeze all layers before the `fine_tune_at` layer
#for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
#
#model.compile(loss=loss,
#              optimizer=Adam(lr=0.001),
#              metrics=['accuracy'])
#
#model.summary()
#
#
## In[20]:
#
#
#import datetime, os
#
#epochs = 10
#steps_per_epoch = train_generator.n // train_generator.batch_size
#validation_steps = validation_generator.n // validation_generator.batch_size
##steps_per_epoch = 50
##validation_steps = 50
#
#logdir = os.path.join("/data/tf_logs_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                         save_weights_only=True, save_best_only=True,
#                                                         verbose=1)
#history = model.fit_generator(
#      train_generator,
#      steps_per_epoch=steps_per_epoch,
#      validation_data=validation_generator,
#      validation_steps=validation_steps,
#      callbacks=[checkpoint_callback, tensorboard_callback],
#      epochs=epochs)
#
#
#
## In[21]:
#
#
#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#plt.figure(figsize=(8, 8))
#plt.subplot(2, 1, 1)
#plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()), 1])
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 1, 2)
#plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.ylabel('Cross Entropy')
#plt.ylim([0, 1.0])
#plt.title('Training and Validation Loss')
#plt.xlabel('epoch')
#
#
## ### Save Model
#
## In[19]:
#
#
#def export(model, path):
#       model.save(path, save_format='tf')
#
#
## In[20]:
#
#
#model.save('/data/saved_model_2/')
#
#
## ### Reload Model
#
## In[11]:
#
#
#import tensorflow as tf
#model = tf.keras.models.load_model('/data/saved_model_2/')
#
#
## In[ ]:
#
#
#
#

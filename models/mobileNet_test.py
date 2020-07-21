#import tensorflow as tf

#from tensorflow import keras
from keras.applications import MobileNet
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

import os
MODEL_SAVE_FOLDER_PATH = './model/'

#hyper param
batch_size = 32
epochs = 1000
learning_rate = 0.001

MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))

for layer in MobileNet.layers:
	layer.trainable = True

for(i,layer) in enumerate(MobileNet.layers):
	print(str(i),layer.__class__.__name__,layer.trainable)

def addTopModelMobileNet(bottom_model, num_classes):
	top_model = bottom_model.output
	top_model = GlobalAveragePooling2D()(top_model)
	top_model = Dense(1024,activation='relu')(top_model)
	top_model = Dense(1024,activation='relu')(top_model)
	top_model = Dense(512,activation='relu')(top_model)
	top_model = Dense(num_classes,activation='softmax')(top_model)

	return top_model

num_classes = 3 #['normal','jump','sit down']
FC_Head = addTopModelMobileNet(MobileNet,num_classes)

model = Model(inputs=MobileNet.input, outputs=FC_Head)

print(model.summary())	

#data generator
#please use Absolute path
train_data_dir = ' ' #train image directory
valid_data_dir = ' ' #validation image directory

train_dataGenerator = ImageDataGenerator( 
			rescale=1./255,
			rotation_range=30,
			width_shift_range=0.3,
			height_shift_range=0.3,
			horizontal_flip=True,
			fill_mode='nearest')
	
valid_dataGenerator = ImageDataGenerator(resclae=1./255)

train_gen = train_dataGenerator.flow_from_directory(
			train_data_dir,
			target_size=(224,224),
			batch_size=batch_size,	
			class_mode='categorical')

valid_gen = valid_dataGenerator.flow_from_directory(
			valid_data_dir,
			target_size=(224,224),
			batch_size=batch_size,
			class_mode='categorical')


#optimizer and callback,checkpoint func
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
	os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH+'{epoch:02d}-{val_loss:.4f}.hdf5'
cb_checkpoint = ModelCheckpoint(
			filepath=model_path,
			monitor='val_loss',
			verbose=1,
			save_best_only=False)

cb_earlyStopping = EarlyStopping(monitor='val_loss',patience=100)

lr_reduction = ReduceLROnPlateau(
			monitor='val_acc',
			patience=5,
			verbose=1,
			factor=0.2,
			min_lr=0.0001)

callbacks = [cb_checkpoint,cb_earlyStopping,lr_reduction]

#model compile and train
model.complie(loss='categorical_crossentropy',optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

nb_train_data = os.walk('train_data_dir').next()[2]
nb_valid_data = os.walk('valid_data_dir').next()[2]

history = model.fit_genorator(
			train_gen,
			steps_per_epoch=nb_train_data//batch_size,
			epochs=epochs,
			callbacks=callbacks,
			validation_data=valid_gen,
			validation_steps=nb_valid_data//batch_size)

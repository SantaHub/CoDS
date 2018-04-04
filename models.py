from keras import backend as K

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Embedding

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
import os


# For channel Reversing in 3D data input
# 	K.set_image_dim_ordering("th")



################### Normal MODELS ######################

## Input 
 # X.shape ## (1133,1,96,2584)
 # X_re = X.reshape(1133,1,96*2584)

  ## Working LSTM
     # model = Sequential()
     # model.add(LSTM(12, input_shape=(1,248064)) )
     # model.add(Dropout(0.25))
     # model.add(ELU(alpha=0.1)) 
     # model.add(Dense(8) )
     # model.add(Activation('relu'))
     # model.add(Dense(len(class_names)) )
     # model.add( Activation("softmax" ))

 ## Working GRU
     # model = Sequential()
     # model.add(GRU(128,input_shape=(1,248064)) )
     # model.add(Dropout(0.25))
     # model.add(ELU(alpha=0.1)) 
     # model.add(Dense(8))
     # model.add(Activation('relu'))
     # model.add(Dense(len(class_names)) )
     # model.add( Activation("softmax" ))

     # model.compile(loss='categorical_crossentropy',
     #           optimizer='adadelta',
     #           metrics=['accuracy'])
     # model.summary()

 ## Working Conv2D
 	## Give input as 3D array. Dont reshape.
	# model = Sequential()
	# model.add(Conv2D(32, 1,
	#               border_mode='valid', input_shape=input_shape))
	# model.add(BatchNormalization(axis=1))
	# model.add(Activation('relu'))
	# model.add(Flatten())
	# model.add(Dense(128))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(nb_classes))
	# model.add(Activation("softmax"))

	# model.compile(loss='categorical_crossentropy',
	#        optimizer='adadelta',
	#        metrics=['accuracy'])
	# model.summary()


########################### SEXY MODELS ####################

 ## Working CNN Sexy
	# model = Sequential()
	# model.add(Conv2D(nb_filters, 1,
	#                     border_mode='valid', input_shape=input_shape))
	# model.add(BatchNormalization(axis=1))
	# model.add(Activation('relu'))

	# for layer in range(nb_layers-1):
	#     model.add(Conv2D(nb_filters, 1))
	#     model.add(BatchNormalization(axis=1))
	#     model.add(ELU(alpha=1.0))  
	#     model.add(MaxPooling2D(pool_size=1))
	#     model.add(Dropout(0.25))

	# model.add(Flatten())
	# model.add(Dense(128))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(nb_classes))
	# model.add(Activation("softmax"))


## AlexNet
	# from keras.models import Model
	# from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
 #    Input, merge, Lambda
	# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

	# input_shape=(1,96,2584)
 #    inputs = Input(shape=input_shape)
 #    conv_1 = Conv2D(96, 1, 1,subsample=(4,4),activation='relu',
 #        name='conv_1', init='he_normal')(inputs)
 #    conv_2 = MaxPooling2D(( 1, 1), strides=(1,1))(conv_1)
 #    conv_2 = BatchNormalization(axis =1,name="convpool_1")(conv_2)
 #    conv_2 = ZeroPadding2D((2,2))(conv_2)
 #    conv_2 = Conv2D(128,5,5,activation="relu",init='he_normal', name='conv_2_1')(conv_2)
 #    conv_3 = MaxPooling2D((1, 1), strides=(2, 2))(conv_2)
 #    conv_3 = BatchNormalization(axis=1)(conv_3)
 #    conv_3 = ZeroPadding2D((1,1))(conv_3)
 #    conv_3 = Conv2D(384,3,3,activation='relu',name='conv_3_1',init='he_normal')(conv_3)
 #    conv_4 = ZeroPadding2D((1,1))(conv_3)
 #    conv_4 = Conv2D(192,3,3,activation="relu", init='he_normal', name='conv_4_1' )(conv_4)
 #    conv_5 = ZeroPadding2D((1,1))(conv_4)
 #    conv_5 = Conv2D(128,3,3,activation="relu",init='he_normal', name='conv_5_1') (conv_5)
 #    dense_1 = MaxPooling2D((1, 1), strides=(2,2),name="convpool_5")(conv_5)
 #    dense_1 = Flatten(name="flatten")(dense_1)
 #    dense_1 = Dense(4096, activation='relu',name='dense_1',init='he_normal')(dense_1)
 #    dense_2 = Dropout(0.5)(dense_1)
 #    dense_2 = Dense(4096, activation='relu',name='dense_2',init='he_normal')(dense_2)
 #    dense_3 = Dropout(0.5)(dense_2)
 #    dense_3 = Dense(nb_classes,name='dense_3_new',init='he_normal')(dense_3)
 #    prediction = Activation("softmax",name="softmax")(dense_3)
 #    alexnet = Model(input=inputs, output=prediction)

## Capsule Network
	x = layers.Input(shape=input_shape)
	conv1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu', name='conv1')(x)
	# conv1 = Conv2D(filters=256, kernel_size=1, border_mode='valid', input_shape=input_shape)
	primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=1, strides=2, padding='valid')
	# digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
	digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, name='digitcaps')(primarycaps) #Gave default routing =3 due to error.
	y = layers.Input(shape=(n_class,))
	masked_by_y = Mask()([digitcaps, y]) 
	masked = Mask()(digitcaps)
	# Shared Decoder model in training and prediction
	decoder = models.Sequential(name='decoder')
	decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
	decoder.add(layers.Dense(1024, activation='relu'))
	decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
	decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
	# Models for training and evaluation (prediction)
	train_model = models.Model([x, y], [ decoder(masked_by_y)])
	eval_model = models.Model(x, [ decoder(masked)])   
	#Noise 
	noise = layers.Input(shape=(n_class, 16))
 	noised_digitcaps = layers.Add()([digitcaps, noise])
 	masked_noised_y = Mask()([noised_digitcaps, y])
	manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))


########################### Validation ####################

## Create TensorBoard Graphs while Training
	# early_release = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')
	# tbCallback = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Result = model.fit(X_train, Y_train, epochs=20, shuffle=True, batch_size=32, verbose=1, validation_split=0.3, callbacks=[tbCallback, early_release])

########################### Visualization ####################

# summarize model for Accuracy
# plt.plot(Result.history['acc'])
# plt.plot(Result.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper left')
# # plt.savefig('model_name'+clf_type+'_acc.png')
# plt.show()


# summarize model for loss
# plt.plot(Result.history['loss'])
# plt.plot(Result.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper left')
# #plt.savefig('model_name'+clf_type+'_loss.png')
# plt.show()


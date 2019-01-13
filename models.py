from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def model_fn(state_shape, action_size):
	inputs = Input(state_shape)

	x1 = Conv2D(128, (4, 1), kernel_initializer='he_uniform')(inputs)
	x2 = Conv2D(128, (1, 4), kernel_initializer='he_uniform')(inputs)
	x3 = Conv2D(128, (2, 2), kernel_initializer='he_uniform')(inputs)
	x4 = Conv2D(128, (3, 3), kernel_initializer='he_uniform')(inputs)
	x5 = Conv2D(128, (4, 4), kernel_initializer='he_uniform')(inputs)

	x1 = Flatten()(x1)
	x2 = Flatten()(x2)
	x3 = Flatten()(x3)
	x4 = Flatten()(x4)
	x5 = Flatten()(x5)

	x = Concatenate()([x1, x2, x3, x4, x5])
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(512, kernel_initializer='he_uniform')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(128, kernel_initializer='he_uniform')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	outputs = Dense(action_size, activation='linear')(x)

	model = Model(inputs, outputs)
	model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])

	return model

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
num_classes = 100

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 将标签转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def resnet_block(inputs, filters, strides=1, activation='relu'):
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def resnet_layer(inputs, num_filters, num_blocks, strides=1):
    x = resnet_block(inputs, num_filters, strides)
    for _ in range(num_blocks - 1):
        x = resnet_block(x, num_filters, activation=None)
        x = Add()([inputs, x])
        x = Activation('relu')(x)
    return x

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters=64)
    x = resnet_layer(x, num_filters=128, strides=2)
    x = resnet_layer(x, num_filters=256, strides=2)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 构建ResNet模型
input_shape = x_train.shape[1:]
model = build_resnet(input_shape, num_classes)


# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))

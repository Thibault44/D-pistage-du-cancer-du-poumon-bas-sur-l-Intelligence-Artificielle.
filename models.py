from keras import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, GlobalAveragePooling3D, Dense


def get_model(num_classes):
    model = Sequential([
        Conv3D(64, kernel_size=3, activation='relu'),
        Conv3D(64, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        BatchNormalization(),

        Conv3D(32, kernel_size=3, activation='relu'),
        Conv3D(32, kernel_size=3, activation='relu'),
        MaxPooling3D(pool_size=2),
        BatchNormalization(),


        GlobalAveragePooling3D(),

        Dense(num_classes, activation='softmax')
    ])

    return model

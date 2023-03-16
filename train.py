import time
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from sklearn import preprocessing

from modeling.models import get_model


def train_preprocessing(array, label):
    array = tf.expand_dims(array, axis=3)
    return array / 255, label


def test_preprocessing(array, label):
    array = tf.expand_dims(array, axis=3)
    return array / 255, label


def load_data():
    annotations_path = '/content/entretien/data/annotations.csv'
    annotations_df = pd.read_csv(annotations_path)
    
    #On procede à l'encode de la variable categorielle density
    encod = preprocessing.OrdinalEncoder(categories='auto')
    encod.fit(annotations_df['density'].values.reshape(-1,1))
    annotations_df['density'] = encod.transform(pd.DataFrame(annotations_df.density.values.reshape(-1,1)))

    train_df = annotations_df.sample(frac=0.8)
    test_df = annotations_df[~annotations_df['nodule_id'].isin(train_df['nodule_id'])]

    nodule_ids_train = train_df['nodule_id']
    nodule_ids_test = test_df['nodule_id']
    

    nodule_arrays_train = [np.load(f'/content/entretien/data/nodules/{nod_id}.npy') for nod_id in nodule_ids_train]
    nodules_train = np.array(nodule_arrays_train)

    nodule_arrays_test = [np.load(f'/content/entretien/data/nodules/{nod_id}.npy') for nod_id in nodule_ids_test]
    nodules_test = np.array(nodule_arrays_test)

    labels_train = to_categorical(train_df['density'].to_numpy())
    labels_test = to_categorical(test_df['density'].to_numpy())

    return nodules_train, labels_train, nodules_test, labels_test

def get_datasets(x_train, y_train, x_test, y_test):
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64

    train_dataset = (
        train_loader.shuffle(len(x_train))
            .map(train_preprocessing)
            .batch(batch_size)
    )

    test_dataset = (
        test_loader.shuffle(len(x_test))
            .map(test_preprocessing)
            .batch(batch_size)
    )

    return train_dataset, test_dataset


def train(model, train_dataset, test_dataset):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['Precision', 'Recall']
    )

    run_id = time.strftime("texture_run_%Y_%m_%d-%H_%M_%S")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        join('/content/entretien/checkpoints', run_id + '.h5'), save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8)
    lrate_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=1)
    epochs = 1000
    history=model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, lrate_scheduler]
    )
    affichage(history)

def get_confmat(model, test_dataset, y_test):
    y_test_cat = np.argmax(y_test, axis=1)

    y_pred = model.predict(test_dataset)
    y_pred_cat = np.argmax(y_pred, axis=1)

    return confusion_matrix(y_test_cat, y_pred_cat)

#fonction permettant l'affiche de l'accuracy et du loss
def affichage(history):
    pr = history.history['precision']
    val_pr= history.history['val_precision']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(pr)), pr, label='Training precision')
    plt.plot(range(len(pr)), val_pr, label='Validation precision')
    plt.legend(loc='lower right')
    plt.title('Training and Validation precision')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(pr)), loss, label='Training Loss')
    plt.plot(range(len(pr)), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
def main():
    x_train, y_train, x_test, y_test = load_data()
    train_dataset, test_dataset = get_datasets(x_train, y_train, x_test, y_test)
    input_shape = (None, 32, 32, 32, 1)
    model = get_model(num_classes=3)
    model.build(input_shape=input_shape)
    model.summary()
    train(model, train_dataset, test_dataset)

    print(get_confmat(model, test_dataset, y_test))


if __name__ == '__main__':
    main()

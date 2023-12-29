import numpy as np
import os
import json
import soundfile as sf
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
import matplotlib.pyplot as plt

#%%
def train_model(X_train, Y_train, vb=2):

    ### Hyper parameters ###
    batch_size = 64
    num_epochs = 512
    kernel_size = 9
    pool_size = 4
    conv_depth_1 = 32
    conv_depth_2 = 64
    conv_depth_3 = 128
    drop_prob_1 = 0.5
    drop_prob_2 = 0.5
    drop_prob_3 = 0.5
    hidden_size = 128

    ### generating the network ###
    model = Sequential()

    # 1st conv block #
    model.add(Convolution1D(conv_depth_1, kernel_size = kernel_size, input_shape=(1001, 1), \
                            padding='same', activation='relu'))
    model.add(Convolution1D(conv_depth_1, kernel_size = kernel_size, \
                            padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Dropout(drop_prob_1))

    # 2nd conv block #
    model.add(Convolution1D(conv_depth_2, kernel_size=kernel_size, \
                            padding='same', activation='relu'))
    model.add(Convolution1D(conv_depth_2, kernel_size=kernel_size, \
                            padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(Dropout(drop_prob_1))

    # 3rd conv block #
    if conv_depth_3 > 0:
        model.add(Convolution1D(conv_depth_3, kernel_size=kernel_size, \
                                padding='same', activation='relu'))
        model.add(Convolution1D(conv_depth_3, kernel_size=kernel_size, \
                                padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size = pool_size))
        model.add(Dropout(drop_prob_3))

    # Dense layers #
    model.add(Flatten())
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dropout(drop_prob_2))
    model.add(Dense(1, activation='sigmoid'))

    # Complie #
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # Fit #
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    print(class_weights)
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=vb,
                        class_weight={0: class_weights[0], 1: class_weights[1]},
                        callbacks=[tf.keras.callbacks.History()])

    return model, history

def load_dataset(filenames):
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  return dataset

def read_audio_files(directory,y):
    audio_files = []
    y_lebal=[]
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav'):
                wavfile=os.path.join(dirpath,filename)
                wavsignal, rt = sf.read(wavfile)
                audio_files.append(list(wavsignal))
                y_lebal.append(y)
    return audio_files,y_lebal

def main():
    filenames = "./PlantSounds/"
    cla_dict = {'Tomato Cut': 0, 'Tomato Dry': 1}

    directories = ['Tomato Cut', 'Tomato Dry']
    all_audio_files = []
    all_y = []
    for directory in directories:
        print(filenames + directory)
        audio, y_label = read_audio_files(filenames + directory, cla_dict[directory])
        all_audio_files.extend(audio)
        all_y.extend(y_label)
    all_audio_files = np.array(all_audio_files)
    all_y = np.array(all_y)

    # Reshape all_audio_files to a 3D array (3000, 1001, 1)
    all_audio_files = np.expand_dims(all_audio_files, axis=-1)

    print(all_audio_files.shape)
    print(all_y.shape)

    model, history = train_model(all_audio_files, all_y)

    # Print the output shape of the CNN model
    output_shape = model.layers[-1].output_shape
    print("CNN model output shape:", output_shape)

    t_loss, t_acc = model.evaluate(all_audio_files, all_y)
    print('训练模型的损失值:', t_loss)
    print('训练模型的准确度:', t_acc)

    # Visualization
    plot_loss_and_accuracy(history)

def plot_loss_and_accuracy(history):
    loss = history.history['loss']
    acc = history.history['binary_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout(h_pad=0.232)
    plt.show()

if __name__ == '__main__':
    main()

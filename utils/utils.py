import numpy as np
import os, sys
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import tensorflow as tf
import pandas as pd
import shap
from keras import applications, optimizers, Input
from tensorflow.keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import load_model


folder = '/home/bvpdsilva//MRI_Brain_Tumor/Train/'
print(folder)

image_width = 128
image_height = 128
channels = 1

train_files = []
i = 0
for mri in ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, mri + '/images'))if os.path.isfile(os.path.join(folder, mri + '/images', f))]
    for _file in onlyfiles:
        train_files.append(_file)

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
y_dataset = []

i = 0
for mri in ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, mri + '/images'))if os.path.isfile(os.path.join(folder, mri + '/images', f))]
    for _file in onlyfiles:
        img_path = os.path.join(folder, mri + '/images', _file)
        img = load_img(img_path, target_size=(image_height, image_width), color_mode='grayscale')
        x = img_to_array(img)
        dataset[i] = x
        mapping = {'Glioma' : 0, 'Meningioma' : 1, 'No Tumor' : 2, 'Pituitary' : 3}
        y_dataset.append(mapping[mri])
        i += 1
        if i == 30000:
            print("%d images to array" % i)
            break
print("All images to array!")

dataset = dataset.astype('float32')
dataset /= 255

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituiary']

first_image_index = {}

for i, label in enumerate(y_dataset):
    if label not in first_image_index:
        first_image_index[label] = i

num_classes = len(set(y_dataset))
num_images_per_class = 1
num_cols = num_classes
num_rows = num_images_per_class

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

for i in range(num_classes):
    idx = first_image_index[i]

    pixels = dataset[idx].reshape(image_height, image_width)

    axes[i].imshow(pixels, cmap='gray')
    axes[i].axis('off')

    axes[i].set_title(f'{classes[i]}')

# Exibe a figura
plt.tight_layout()
plt.show()

nclasses = len(set(y_dataset))
print(num_classes)

y_dataset_ = to_categorical(y_dataset, nclasses)

dataset_trimmed = dataset[:len(y_dataset_)]

x_train, x_test, y_train, y_test = train_test_split(dataset_trimmed, y_dataset_, test_size=0.2)

print("Train set size: {0}, Test set size: {1}".format(len(x_train), len(x_test)))

balanced_x_train = []
balanced_y_train = []

majority_samples = 500

for class_label in np.unique(y_train.argmax(axis=1)):
    x_class = x_train[y_train.argmax(axis=1) == class_label]
    y_class = y_train[y_train.argmax(axis=1) == class_label]

    minority_samples = len(x_class)
    balanced_x_class, balanced_y_class = resample(x_class, y_class,
                                                  replace=True,
                                                  n_samples=majority_samples,
                                                  random_state=42)
    balanced_x_train.extend(balanced_x_class)
    balanced_y_train.extend(balanced_y_class)

balanced_x_train = np.array(balanced_x_train)
balanced_y_train = np.array(balanced_y_train)

shuffled_indices = np.arange(len(balanced_x_train))
np.random.shuffle(shuffled_indices)
balanced_x_train = balanced_x_train[shuffled_indices]
balanced_y_train = balanced_y_train[shuffled_indices]

print("Tamanho do conjunto de treinamento balanceado:", len(balanced_x_train))
print("Tamanho do conjunto de teste:", len(x_test))

for class_label in np.unique(balanced_y_train.argmax(axis=1)):
    count = np.sum(balanced_y_train.argmax(axis=1) == class_label)
    print(f"Classe {class_label}: {count} amostras")

model = Sequential()

model.add(BatchNormalization(input_shape=(image_height, image_width, 1)))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(4, activation='softmax'))

model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,  restore_best_weights=True)
history = model.fit(balanced_x_train, balanced_y_train, validation_split= 0.2, batch_size=32, shuffle=True, epochs=50, callbacks=[early_stopping])

history_salvo = pd.DataFrame(history.history)
history_salvo.to_csv('history_salvo90valAcc.csv')

model_json = model.to_json()
with open("MRI_modelcnn90valAcc.json", "w") as json_file:
    json_file.write(model_json)

pd.DataFrame(history.history).to_csv('loss.csv', index=False)
model.save('modelo_cnn90valAcc.h5')

modelo_carregado = load_model('/home/bvpdsilva/MRI_Brain_Tumor/utils/modelo_cnn90valAcc.h5')

val_accuracy = history.history['val_accuracy']
mean_val_accuracy = np.mean(val_accuracy)
print("Média da validação de acurácia:", mean_val_accuracy)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'])
plt.savefig('loss_plot.png')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend(['Treinamento', 'Validação'])
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()
plt.close()

preds = modelo_carregado.predict(x_test)

def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues
    ):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe real')
    plt.xlabel('Classe predita')

y_test_ = [np.argmax(x) for x in y_test]
preds_ = [np.argmax(x) for x in preds]

cm = confusion_matrix(y_test_, preds_)
plot_confusion_matrix(cm, classes=['Glioma', 'Meningloma', 'No Tumor', 'Pituitary'], title='Confusion matrix')
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()

# Calcular acurácia
accuracy = accuracy_score(y_test_, preds_)
print("Acurácia:", accuracy*float(100.0), "%")

# Calcular precisão
precision = precision_score(y_test_, preds_, average='macro')
print("Precisão:", precision*float(100.0), "%")

# Calcular recall
recall = recall_score(y_test_, preds_, average='macro')
print("Recall:", recall*float(100.0), "%")

# Calcular F1 score
f1 = f1_score(y_test_, preds_, average='macro')
print("F1-score:", f1*float(100.0), "%")

n = 4
for t in range(4):
    plt.figure(figsize=(10,10))
    for i in range(n*t, n*(t+1)):
        plt.subplot(1, n, i + 1 - n*t)
        plt.imshow(cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Real: {}\nPredito: {}'.format(classes[np.argmax(y_test[i])], classes[np.argmax(preds[i])]))
        plt.axis('off')
    plt.savefig(f'test_predictions_{t}.png')
    plt.show()

def normalize_shap_values(shap_values, epsilon=1e-8):
    normalized_shap = []
    for val in shap_values:
        min_val = np.min(val)
        max_val = np.max(val)
        # Evitar divisão por zero
        range_val = max_val - min_val + epsilon
        normalized_shap.append((val - min_val) / range_val)  # Normalizar para o intervalo [0, 1]
    return np.array(normalized_shap)

# Implementação do SHAP
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
explainer = shap.GradientExplainer(model, background)

# Explicar as previsões para as primeiras 10 imagens de teste
shap_values = explainer.shap_values(x_test[:10])

# Normalizar os valores SHAP para o intervalo [0, 1]
normalized_shap_values = normalize_shap_values(shap_values)

# Títulos personalizados para as imagens
titles_left = [f"Resultado: {'Tumor' if y_test[i] == 1 else 'Normal'}" for i in range(10)]
titles_right = ["Pixels vistos pela IA" for _ in range(len(normalized_shap_values))]

# Plotar os valores SHAP para as primeiras 10 imagens de teste
plt.figure(figsize=(15, 5))
shap.image_plot(normalized_shap_values, x_test[:10], show=False)

# Acessar os subplots (imagens da esquerda e direita para personalização)
axes = plt.gcf().axes
for i in range(0, len(titles_left) * 2, 2):
    axes[i].set_title(titles_left[i // 2], fontsize=12, color="darkblue")

for i in range(1, len(titles_right) * 2, 2):
    axes[i].set_title(titles_right[i // 2], fontsize=12, color="darkred")

# Título geral e legenda explicativa
plt.suptitle('Explicabilidade para Classificação de Tumores', fontsize=16)
plt.figtext(0.5, 0.01, "Cores mais claras indicam maior impacto positivo (rosa) e negativo (azul) na previsão", ha="center", fontsize=12)

plt.show()

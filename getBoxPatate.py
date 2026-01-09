
'''
modele keras : classification d'image
nécessite beaucoup d'adaptation
pas pratique
'''
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, datasets

#class_names = ['classe1', 'classe2', 'classe3', ...]  # Remplace par tes classes
class_names = ['patate' , 'carotte']

###redimmensionner les images pour compatibilité avec le modele
def resize_images(images, target_size=(64, 64)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return np.array(resized_images)
'''
chatgement des images et annotations depuis le repertoire dataset
'''
def load_images_from_folder(folder):
    images = []
    train_image_files = sorted(os.listdir(folder))
    for filename in train_image_files:
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            print('fichier jpg: {}'.format(filename))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
            #img = cv2.resize(img, (224 , 224))
            img = img / 255.0  # Normalisation
            images.append(img)
    return np.array(images)
import xml.etree.ElementTree as ET
##lire les annotations des images précédentes
def load_annotations_from_folder(folder, class_names):
    print('classe name passée en paramtre: {}'.format(class_names))
    labels = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.xml'):
            print('fichier xml: {}'.format(filename))
            tree = ET.parse(os.path.join(folder, filename))
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in class_names:  # Vérifie que la classe existe dans class_names
                    label = class_names.index(class_name)
                    labels.append(label)
                    break  # Prend seulement la première classe si plusieurs objets
    return np.array(labels)


'''
charger les train et test pour entrainer le modele
'''
# Chemin vers tes dossiers
train_images_dir = "mondataset/train/images"
train_annotations_dir = "mondataset/train/annotations"
test_images_dir = "mondataset/test/images"
test_annotations_dir = "mondataset/test/annotations"

# Charger les images et les labels
train_images = load_images_from_folder(train_images_dir)
train_labels = load_annotations_from_folder(train_annotations_dir, class_names)
test_images = load_images_from_folder(test_images_dir)
test_labels = load_annotations_from_folder(test_annotations_dir, class_names)
#redimensionner les images en 64*64
taille_image = 32
train_images = resize_images(train_images, (taille_image, taille_image))
test_images = resize_images(test_images, (taille_image, taille_image))
#verification des chargements
print('Train: nb img={}\tnb lab={}'.format(len(train_images) , len(train_labels)))
print('Test : nb img={}\tnb lab={}'.format(len(test_images) , len(test_labels)))
print("Train labels:", train_labels)
print("Test labels:", test_labels)
print("Train images shape:", train_images.shape)
print("Train images min/max:", train_images.min(), train_images.max())
#plt.imshow(train_images[0])
#plt.show()
# Affiche les labels uniques pour vérifier
print("Train labels uniques:", np.unique(train_labels))
print("Test labels uniques:", np.unique(test_labels))
#mettre a jour le modele ( creation + enchainement des couches
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(taille_image, taille_image, 3)),  # Mettre à jour input_shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(taille_image, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(taille_image, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(taille_image, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Mettre à jour le nombre de classes
])
'''from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(taille_image, taille_image, 3))
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    #layers.Dropout(0 , 5),
    #layers.Dense(2, activation='sigmoid'),
    #layers.Dense(2, activation='sparse_categorical_crossentropy'),
    layers.Dense(len(class_names), activation='softmax')
])'''


##compiler et entrainer le modele sur le dataset
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#auto generation d'images pour augmenter le dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True , zoom_range=0.2)
datagen.fit(train_images)
#model.fit(datagen.flow(train_images, train_labels, batch_size=8), epochs=10)
history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
#verification de l'apprrentissage
'''plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()'''

#epoch=10
#model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))

#evaluer le modele
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Précision sur le jeu de test : {test_acc:.4f}")

#test de predictions sur le jeu de tests
predictions = model.predict(test_images)
print("Predictions:", np.argmax(predictions, axis=1))
print("True labels:", test_labels)
model.save("mon_modele", save_format="tf")  # Crée un dossier "mon_modele"
'''Programme d' analyse en temps réel'''        'rtsp://admin:@192.168.1.26/play1.sdp'

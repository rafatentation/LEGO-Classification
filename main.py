import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

dataset_dir = './DATASET' 
img_width, img_height = 128, 128
batch_size = 32

print(f"Veri yolu: {dataset_dir}")


train_datagen = ImageDataGenerator(
    rescale=1./255,            
    rotation_range=20,         
    width_shift_range=0.1,      
    height_shift_range=0.1,     
    shear_range=0.1,           
    zoom_range=0.1,             
    horizontal_flip=True
    fill_mode='nearest',       
    validation_split=0.2        
)


train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


labels = (train_generator.class_indices)
print(f"Sınıflar: {labels}")
num_classes = len(labels)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Eğitim başlıyor...")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[early_stop]
)


model.save('gelismis_lego_modeli.h5')
print("Model kaydedildi.")

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.title('Model Doğruluk Oranı')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.title('Model Kayıp Oranı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.savefig('basari_grafigi.png')
print("Grafik 'basari_grafigi.png' olarak kaydedildi.")
plt.show() 
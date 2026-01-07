import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. AYARLAR ---
# Klasör adını kendine göre düzenle
dataset_dir = './DATASET' 
img_width, img_height = 128, 128  # Resim boyutunu değiştirdik (daha hızlı işlem için)
batch_size = 32

print(f"Veri yolu: {dataset_dir}")

# --- 2. GELİŞMİŞ VERİ İŞLEME (DATA AUGMENTATION) ---
# Önceki koddan FARKLI olarak: Resimleri sadece okumuyoruz, 
# onları döndürüp, yakınlaştırıp, çevirip veri setini yapay olarak çoğaltıyoruz.
# Bu, LEGO gibi az verili setlerde başarıyı çok artırır.

train_datagen = ImageDataGenerator(
    rescale=1./255,             # Piksel normalizasyonu
    rotation_range=20,          # Resmi 20 derece döndür
    width_shift_range=0.1,      # Sağa sola kaydır
    height_shift_range=0.1,     # Yukarı aşağı kaydır
    shear_range=0.1,            # Resmi hafifçe bük
    zoom_range=0.1,             # Yakınlaştır
    horizontal_flip=True,       # Yatay çevir
    fill_mode='nearest',        # Boşlukları doldur
    validation_split=0.2        # %20 Test için ayır
)

# Eğitim Verisi
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Test Verisi (Test verisine augmentation uygulanmaz, sadece rescale)
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Sınıf isimlerini yazdıralım
labels = (train_generator.class_indices)
print(f"Sınıflar: {labels}")
num_classes = len(labels)

# --- 3. MODEL MİMARİSİ (Daha Derin ve Farklı) ---
model = Sequential()

# 1. Blok
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization()) # FARK: Öğrenmeyi hızlandırır ve stabilize eder
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2. Blok
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization()) # Her katmana ekledik
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3. Blok
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4. Blok (Daha derin özellikler için)
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Sınıflandırma Katmanı
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # Ezberlemeyi önlemek için %50 nöronu kapat
model.add(Dense(num_classes, activation='softmax')) # Çoklu sınıflandırma çıkışı

# Derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. EĞİTİM VE ERKEN DURDURMA ---
# FARK: EarlyStopping ekledik. Eğer model gelişmeyi durdurursa eğitimi otomatik keser.
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Eğitim başlıyor...")
history = model.fit(
    train_generator,
    epochs=15, # Epoch sayısını artırdık, early_stop gerekirse durduracak
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# --- 5. KAYDETME VE GRAFİK ÇİZDİRME ---
model.save('gelismis_lego_modeli.h5')
print("Model kaydedildi.")

# Başarı Grafiğini Çiz (Raporun için harika olur)
plt.figure(figsize=(12, 4))

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.title('Model Doğruluk Oranı')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.title('Model Kayıp Oranı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

# Grafiği kaydet ve göster
plt.savefig('basari_grafigi.png')
print("Grafik 'basari_grafigi.png' olarak kaydedildi.")
plt.show() 
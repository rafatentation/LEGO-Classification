# Sinir Ağları Dersi Final Projesi: LEGO Parça Sınıflandırma

Bu repo, **Sinir Ağları** dersi kapsamında hazırlanan final projesine aittir. Projede, Kaggle üzerinden alınan LEGO parça görselleri kullanılarak bir Derin Öğrenme (CNN) modeli eğitilmiştir. Model, farklı LEGO parçalarını otomatik olarak sınıflandırmayı amaçlamaktadır. Tüm kodlar, model mimarisi, eğitim süreci ve sonuçlar bu README dosyasında açıklanmaktadır.

## Proje Ekibi
- **RAUF NURIYEV**

## Veri Seti Açıklaması

**Veri Seti Adı:** LEGO Parts Dataset  
**Kaynak:** https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification?select=test
**Açıklama:**  
Bu veri seti, farklı türdeki LEGO parçalarının görsellerini içermektedir. Her sınıf belirli bir LEGO parça türünü temsil etmektedir. Veri seti, modelin farklı açı, ışık ve pozisyonlardaki LEGO parçalarını tanımayı öğrenmesi için kullanılmıştır.

**Özellikler:**
- Giriş: 128x128 piksel RGB renkli görseller
- Çıkış: Çoklu sınıflandırma (16 sınıf)
- Veri Augmentation: Döndürme, kaydırma, yakınlaştırma, çevirme ile veri çeşitliliği artırılmıştır
- Veri Dağılımı: %80 eğitim, %20 validation split

## Model Mimarisi

Kullanılan CNN modeli aşağıdaki katmanlardan oluşmaktadır:

### 1. Giriş Katmanı
- **Input Shape:** (128, 128, 3) - RGB görseller

### 2. 1. Konvolüsyon Bloku
- **Conv2D:** 32 filtre, (3×3 kernel), ReLU aktivasyon
- **BatchNormalization:** Gamma (32), Beta (32), Moving Mean (32), Moving Variance (32)
- **MaxPooling2D:** (2×2) pooling

### 3. 2. Konvolüsyon Bloku
- **Conv2D:** 64 filtre, (3×3 kernel), ReLU aktivasyon
- **BatchNormalization:** Gamma (64), Beta (64), Moving Mean (64), Moving Variance (64)
- **MaxPooling2D:** (2×2) pooling

### 4. 3. Konvolüsyon Bloku
- **Conv2D:** 128 filtre, (3×3 kernel), ReLU aktivasyon
- **BatchNormalization:** Gamma (128), Beta (128), Moving Mean (128), Moving Variance (128)
- **MaxPooling2D:** (2×2) pooling

### 5. 4. Konvolüsyon Bloku
- **Conv2D:** 256 filtre, (3×3 kernel), ReLU aktivasyon
- **BatchNormalization:** Gamma (256), Beta (256), Moving Mean (256), Moving Variance (256)
- **MaxPooling2D:** (2×2) pooling

### 6. Sınıflandırma Katmanları
- **Flatten:** 3D özellik haritalarını 1D vektöre dönüştürme
- **Dense:** 512 nöron, ReLU aktivasyon
- **Dropout:** 0.5 (Aşırı öğrenmeyi önlemek için)
- **Dense:** 16 nöron, Softmax aktivasyon (16 sınıf için)

**Toplam Parametre:** [model.summary() çıktısına göre otomatik]
**Eğitilebilir Parametre:** [model.summary() çıktısına göre otomatik]

**Optimizasyon Algoritması:** Adam  
**Kayıp Fonksiyonu:** Categorical Crossentropy  
**Batch Size:** 32  
**Epoch:** 15 (EarlyStopping ile optimize edildi)  

## Eğitim Süreci

### Eğitim Stratejileri:
1. **EarlyStopping:** `monitor='val_loss', patience=3, restore_best_weights=True`
2. **Data Augmentation:**
   - Rotation: 20°
   - Width/Height Shift: %10
   - Shear: %10
   - Zoom: %10
   - Horizontal Flip: True
3. **BatchNormalization:** Tüm konvolüsyon katmanlarından sonra

### Eğitim Grafikleri
<img width="1200" height="400" alt="basari_grafigi" src="https://github.com/user-attachments/assets/614204a4-d0c4-4fb3-8479-d05d46270c24" />

#### Grafik 1: Model Doğruluk Oranı (Accuracy)
![Model Doğruluk Oranı]

*Grafik Analizi:*
- **Eğitim Başarısı:** Modelin eğitim verisi üzerindeki doğruluk oranı ~%95'e ulaşmıştır
- **Test Başarısı:** Modelin validation verisi üzerindeki doğruluk oranı ~%92 seviyesindedir
- **Değerlendirme:** Eğitim ve test doğrulukları arasındaki küçük fark (~%3), modelin iyi genelleme yaptığını göstermektedir

#### Grafik 2: Model Kayıp Oranı (Loss)
![Model Kayıp Oranı]

*Grafik Analizi:*
- **Eğitim Kaybı:** 0.25 seviyesine düşmüştür
- **Test Kaybı:** 0.30 seviyesinde stabilize olmuştur
- **Değerlendirme:** Kayıp değerlerinin düzenli düşüşü ve stabilizasyonu, modelin sağlıklı öğrendiğini göstermektedir

## Başarı Metrikleri

Model eğitimi tamamlandıktan sonra elde edilen son metrikler:

- **Son Epoch Test Doğruluğu (val_accuracy):** %92
- **Son Epoch Test Kaybı (val_loss):** 0.30
- **Eğitim Doğruluğu (accuracy):** %95
- **Eğitim Kaybı (loss):** 0.25

**Model Performans Değerlendirmesi:**
1. Model 4. epoch'tan sonra kararlı performans göstermiştir
2. EarlyStopping 7-8. epoch'ta eğitimi durdurmuştur
3. Overfitting minimal düzeydedir (%3 fark)
4. BatchNormalization sayesinde öğrenme hızlı ve stabil gerçekleşmiştir

## Netron ile Model Görselleştirme

Modelin yapısı Netron aracılığıyla görselleştirilmiştir:

<img width="181" height="663" alt="NetronGrafik" src="https://github.com/user-attachments/assets/ff9c81e8-0ab7-4a00-9f32-87a03f836ece" />
<img width="132" height="335" alt="NetronGrafik2" src="https://github.com/user-attachments/assets/f5058577-9f0a-460b-b07c-9a418e7df57e" />




### Netron Görseli Detaylı Açıklaması:

#### 1. Giriş ve İlk Katmanlar (İlk Görsel):
- **Input Layer:** (128, 128, 3) boyutunda RGB giriş
- **İlk Conv2D:** 
  - Kernel: (3×3×3×32) - 3 kanal giriş, 32 filtre çıkış
  - Bias: 32 parametre
- **Activation:** ReLU aktivasyon fonksiyonu
- **BatchNormalization:** 
  - Gamma: 32 parametre (ölçeklendirme)
  - Beta: 32 parametre (kaydırma)
  - Moving Mean: 32 parametre (ortalama)
  - Moving Variance: 32 parametre (varyans)
- **MaxPooling2D:** (2×2) boyut indirgeme

#### 2. Orta Katmanlar (İlk Görsel Devamı):
- **İkinci Conv2D:** 
  - Kernel: (3×3×32×64) - 32 kanal giriş, 64 filtre çıkış
  - Bias: 64 parametre
- **BatchNormalization:** 64'er parametre ile gamma, beta, mean, variance
- **Üçüncü Conv2D:**
  - Kernel: (3×3×64×128) - 64 kanal giriş, 128 filtre çıkış
  - Bias: 128 parametre
- **BatchNormalization:** 128'er parametre

#### 3. Çıkış Katmanları (İkinci Görsel):
- **Dördüncü Conv2D:** 256 filtre
- **Flatten Katmanı:** 3D'den 1D'ye dönüşüm
- **Dense Katmanı:** 512 nöron
- **Dropout:** %50 dropout oranı
- **Çıkış Katmanı:** 16 nöron (16 sınıf için Softmax)

**Teknik Detaylar:**
- Her Conv2D katmanından sonra BatchNormalization uygulanmıştır
- Toplam 4 konvolüsyon bloğu bulunmaktadır
- Filtre sayıları: 32 → 64 → 128 → 256 şeklinde katlanarak artmaktadır
- Her pooling katmanı ile özellik haritası boyutu yarıya inmektedir

## Kullanım

### Gereksinimler:
```bash
pip install -r requirements.txt

Modeli Eğitmek:
python main.py

Bu komut:

Veri setini yükler ve augmentation uygular

Modeli oluşturur ve eğitir

Eğitim grafiklerini kaydeder

Modeli gelismis_lego_modeli.h5 olarak kaydeder

Model Dosyası Notu:
gelismis_lego_modeli.h5 dosyası boyutu nedeniyle bu repoda bulunmamaktadır. Ancak main.py dosyasını çalıştırarak modeli sıfırdan eğitebilir ve aynı dosyayı oluşturabilirsiniz. Eğitim yaklaşık 15-20 dakika sürmektedir.
LEGO_CNN_Project/
│
├── archive/                              # LEGO veri seti (Kaggle'dan indirilmeli)
│   ├── 3001/                             # Sınıf 1: 2x4 Brick
│   ├── 3002/                             # Sınıf 2: 2x3 Brick
│   └── ...                               # Diğer LEGO parça sınıfları
│
├── main.py                               # Ana eğitim scripti
├── requirements.txt                      # Python bağımlılıkları
├── gelismis_lego_modeli.h5              # Eğitilmiş model (oluşturulacak)
├── basari_grafigi.png                   # Eğitim grafikleri
├── Ekran-şəkli-2026-01-06-163459.png    # Netron görseli - Bölüm 1
├── Ekran-şəkli-2026-01-06-163524.png    # Netron görseli - Bölüm 2
└── README.md                            # Bu dosya


Sonuç ve Değerlendirme
Başarılı Yönler:
Yüksek Doğruluk: %92 test doğruluğu ile başarılı sınıflandırma

İyi Genelleme: Eğitim-test farkının düşük olması

Modern Teknikler: BatchNormalization, Data Augmentation, EarlyStopping

Derin Mimarİ: 4 katmanlı CNN ile kompleks özellikler öğrenilmiştir

Teknik Katkılar:
BatchNormalization: Öğrenmeyi hızlandırmış ve stabilize etmiştir

Data Augmentation: Sınırlı veriyle daha robust model oluşturulmuştur

EarlyStopping: Overfitting riski minimize edilmiştir

Dropout: Regularization sağlanmıştır

Pratik Uygulama Alanları:
LEGO Fabrikaları: Otomatik kalite kontrol ve sınıflandırma

LEGO Kütüphaneleri: Parça envanter yönetimi

E-ticaret: LEGO parça tanıma sistemleri

Eğitim: Görsel tanıma eğitim materyali

Zorluklar ve Çözümler
Zorluk	Çözüm
Sınırlı veri seti	Data Augmentation teknikleri
Overfitting riski	Dropout + EarlyStopping
Eğitim süresi	BatchNormalization ile hızlandırma
Model karmaşıklığı	4 bloklu optimize mimari
Gelecek Çalışmalar
Transfer Learning: ResNet, VGG gibi önceden eğitilmiş modeller

Hyperparameter Tuning: Grid search ile optimal parametreler

Daha Büyük Veri Seti: Daha fazla LEGO parça çeşidi

Real-time Sınıflandırma: Web kamerası ile canlı test





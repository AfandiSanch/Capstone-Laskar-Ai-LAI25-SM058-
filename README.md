# Proyek Klasifikasi Gambar Sampah Menggunakan CNN

## Deskripsi Proyek
Proyek ini merupakan bagian dari capstone yang bertujuan untuk mengembangkan model klasifikasi gambar menggunakan Convolutional Neural Network (CNN). Model dilatih untuk mengenali dan mengklasifikasikan gambar sampah ke dalam 5 kategori berbeda. Dataset awal memiliki 6 kategori, namun salah satu kategori ("trash") dikeluarkan karena mengandung data yang ambigu.

---

## Tahapan Proyek

### 1. Data Understanding
- Dataset terdiri dari **6 kelas awal**: cardboard, glass, metal, paper, plastic, dan trash.
- Ukuran gambar **seragam** di seluruh kelas.
- **Distribusi data relatif seimbang**, namun kelas **"trash" memiliki keanekaragaman konten yang tinggi** dan mengandung gambar yang seharusnya masuk ke kategori lain.
- Oleh karena itu, kelas **"trash" dikeluarkan** dari proses pelatihan model.

### 2. Data Preparation
- **Kelas "trash" dihapus** dari dataset.
- Dataset dibagi menggunakan `train_test_split` dengan **rasio 80% data latih** dan **20% data uji**:
  - **Training set:** 1912 gambar
  - **Testing set:** 478 gambar
- Sebelum proses pelatihan dilakukan:
  - Disiapkan data generator untuk training, validation, dan testing menggunakan `ImageDataGenerator`.
  - **20% dari data latih** disisihkan sebagai data **validasi**.
  - Parameter penting yang digunakan dalam `ImageDataGenerator`:
    - `target_size`: (256, 256)
    - `batch_size`: 32
    - `color_mode`: 'rgb'
    - `class_mode`: 'sparse'
    - `shuffle`: True (hanya pada data latih)

### 3. Modeling
- **Boosting prioritas kelas** dilakukan pada "glass" dan "metal" sebelum pelatihan untuk mengatasi ketidakseimbangan minor.
- Model CNN dibangun dengan:
  - **3 blok konvolusi** dilanjutkan dengan max pooling.
  - **Beberapa dropout layer** untuk mencegah overfitting.
  - Output layer menggunakan **softmax** untuk klasifikasi **5 kelas**.
- **Early stopping** digunakan untuk menghentikan pelatihan saat performa validasi tidak meningkat lagi, demi efisiensi dan generalisasi yang baik.
- Model dilatih menggunakan **data generator** untuk efisiensi memori.

### 4. Evaluasi Model
- Visualisasi hasil pelatihan dilakukan dengan membuat **plot akurasi dan loss** pada data latih dan validasi.
- Hasil pelatihan pada **epoch ke-25**:
  - Training accuracy: **98.08%**
  - Training loss: **0.0592**
  - Validation accuracy: **59.47%**
  - Validation loss: **1.9887**
- Performa model diukur menggunakan **F1-Score** untuk masing-masing kelas:
  - **Tertinggi:** "cardboard" (0.7190)
  - **Terendah:** "glass" (0.5371)
- **Akurasi total model:** **0.6444** (64.44%)

---

## Kesimpulan
Model CNN mampu mengenali gambar dengan cukup baik di beberapa kelas, tetapi terdapat **overfitting** yang ditandai dengan selisih besar antara akurasi training dan validasi. Performa terbaik diperoleh pada kelas "cardboard", sementara kelas "glass" masih sulit dikenali dengan baik. Rekomendasi untuk pengembangan selanjutnya adalah melakukan **data augmentation** yang lebih agresif dan/atau menggunakan arsitektur CNN yang lebih dalam (seperti ResNet, MobileNet, dll) serta menerapkan **transfer learning**.

---

## Teknologi yang Digunakan
- Python
- TensorFlow & Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- OpenCV, skimage

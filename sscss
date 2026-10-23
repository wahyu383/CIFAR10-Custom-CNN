# Klasifikasi Objek Citra (CIFAR-10) menggunakan Custom CNN 🚀

Repositori ini berisi proyek Ujian Tengah Semester (UTS) mata kuliah **Advance Computer Vision**. Proyek ini berfokus pada pembangunan model *Convolutional Neural Network* (CNN) dari awal (Custom CNN) tanpa menggunakan *transfer learning* untuk mengklasifikasikan 10 kategori objek.

**👤 Identitas Mahasiswa:**
- **Nama:** Wahyu Maulana
- **NIM:** 251012000008

---

## 📌 Deskripsi Proyek
Klasifikasi objek adalah salah satu tugas fundamental dalam *Computer Vision*. Proyek ini bertujuan untuk membangun, melatih, dan mengevaluasi model Custom CNN untuk mendeteksi dan mengklasifikasikan 10 kelas objek (seperti pesawat, mobil, burung, kucing, anjing, dll.) pada dataset CIFAR-10 secara efisien menggunakan *framework* TensorFlow dan Keras.

## 📊 Dataset
Proyek ini menggunakan **Dataset CIFAR-10** yang diunduh secara otomatis melalui *library* TensorFlow/Keras.
- **Total Data:** 60.000 gambar berwarna ukuran 32x32 piksel.
- **Pembagian Data:** 50.000 citra untuk *Training* dan 10.000 citra untuk *Testing*.
- **Kelas:** Terdapat 10 kelas yang seimbang (Pesawat, Mobil, Burung, Kucing, Rusa, Anjing, Katak, Kuda, Kapal, Truk).

## 🛠️ Arsitektur Model (Custom CNN)
Model dibangun menggunakan pendekatan berurutan (*Sequential*) dengan rancangan arsitektur ringan sebagai berikut:
1. **Feature Extraction:** 
   - 3 layer `Conv2D` (kernel 3x3, aktivasi ReLU)
   - 2 layer `MaxPooling2D` (2x2) untuk mereduksi dimensi spasial.
2. **Classifier:** 
   - 1 layer `Flatten` untuk meratakan peta fitur 2D menjadi 1D.
   - 1 layer `Dense` (64 neuron, aktivasi ReLU).
   - 1 layer `Output` (10 neuron, aktivasi Softmax).

## ⚙️ Parameter Pelatihan
- **Optimizer:** Adam (*Adaptive Moment Estimation*)
- **Loss Function:** `sparse_categorical_crossentropy`
- **Epochs:** 10

## 💻 Cara Menjalankan Kode
Karena dataset terintegrasi langsung dengan Keras, Anda tidak perlu mengunduh file gambar secara manual.
1. Unduh file `UTS_Computer_Vision_CIFAR10_Wahyu_Maulana.ipynb`.
2. Buka dan unggah file tersebut ke [Google Colab](https://colab.research.google.com/).
3. Pilih menu **Runtime > Run all** untuk menjalankan proses pelatihan dan evaluasi dari awal hingga akhir.

## 📈 Evaluasi Model
Model dievaluasi menggunakan *Classification Report* (Presisi, Recall, F1-Score) dan divisualisasikan menggunakan:
- Grafik perbandingan *Akurasi* dan *Loss* (Training vs Validation).
- *Confusion Matrix* menggunakan Seaborn untuk melihat sebaran prediksi kelas yang tepat dan keliru.

---
*Dibuat untuk memenuhi tugas UTS Advance Computer Vision - 2026*

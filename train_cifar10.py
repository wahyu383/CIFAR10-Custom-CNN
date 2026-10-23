"""
Klasifikasi Objek Citra (CIFAR-10) menggunakan Custom CNN
Tugas UTS Advance Computer Vision
Nama: Wahyu Maulana
NIM: 251012000008
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def main():
    # ==========================================
    # 1. Memuat dan Preprocessing Data Otomatis
    # ==========================================
    print("Mengunduh dataset CIFAR-10...")
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalisasi nilai piksel menjadi 0 - 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['Pesawat', 'Mobil', 'Burung', 'Kucing', 'Rusa', 
                   'Anjing', 'Katak', 'Kuda', 'Kapal', 'Truk']

    # ==========================================
    # 2. Membangun Arsitektur CNN Sederhana
    # ==========================================
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # ==========================================
    # 3. Kompilasi & Pelatihan Model
    # ==========================================
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Memulai proses pelatihan model...")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    # ==========================================
    # 4. Evaluasi & Visualisasi
    # ==========================================
    print("Menampilkan grafik evaluasi...")
    plt.figure(figsize=(12, 4))
    
    # Grafik Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Grafik Akurasi CNN')
    plt.legend()

    # Grafik Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Grafik Loss CNN')
    plt.legend()
    plt.show()

    # Evaluasi Classification Report & Confusion Matrix
    y_pred = np.argmax(model.predict(test_images), axis=1)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(test_labels, y_pred, target_names=class_names))

    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CIFAR-10')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.show()

if __name__ == "__main__":
    main()

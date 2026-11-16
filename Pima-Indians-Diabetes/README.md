# Prediksi Diabetes dengan Logistic Regression

Proyek ini membangun model klasifikasi untuk memprediksi apakah seorang pasien terdiagnosis diabetes berdasarkan data pengukuran medis, memakai dataset Pima Indians Diabetes.

## Dataset

Dataset berisi 768 data perempuan keturunan Pima berusia minimal 21 tahun, dikumpulkan oleh National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

- **Fitur (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target:** `Outcome` (1 = diabetes, 0 = tidak diabetes)
- **Sebaran kelas:** 268 diabetes dan 500 non-diabetes

> Beberapa kolom (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) berisi nilai 0 yang secara medis tidak masuk akal dan kemungkinan merupakan data hilang. Pada versi saat ini, nilai tersebut belum ditangani.

## Tahapan

1. Memuat data ke dalam DataFrame (`pandas`)
2. Membagi data: 80% latih dan 20% uji, dengan stratifikasi
3. Standarisasi fitur dengan `StandardScaler` (fit hanya pada data latih)
4. Melatih model `LogisticRegression`
5. Evaluasi: akurasi, presisi, recall, F1-score, ROC-AUC, dan confusion matrix

## Hasil (data uji)

| Metrik | Nilai |
|---|---|
| Akurasi | 0,714 |
| Presisi (diabetes) | 0,609 |
| Recall (diabetes) | 0,519 |
| F1-score (diabetes) | 0,560 |
| ROC-AUC | 0,823 |

Model cukup baik membedakan kedua kelas, tetapi recall untuk kelas diabetes masih rendah, sehingga sebagian kasus diabetes belum terdeteksi.

## Cara menjalankan

Library yang dipakai: `pandas`, `scikit-learn`, `matplotlib`, dan `jupyter`.

```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook diabetes_model.ipynb
```

Pastikan `diabetes.csv` berada di folder yang sama dengan notebook.

## Struktur berkas

```
.
├── diabetes_model.ipynb   # notebook utama
├── diabetes.csv           # dataset
├── README.md
└── LICENSE
```

## Rencana pengembangan

- Mengganti nilai 0 yang tidak valid dengan median
- Menambahkan `class_weight="balanced"` untuk meningkatkan recall
- Membandingkan dengan model lain (Random Forest, dsb.) dan menggunakan cross-validation

## Sumber data

Pima Indians Diabetes Database, NIDDK. Cantumkan tempat Anda mengunduh datasetnya (misalnya Kaggle atau UCI Machine Learning Repository).

## Lisensi

Kode dalam repositori ini dilisensikan di bawah [Apache License 2.0](LICENSE).

Copyright [tahun] [Nama Anda]

Dataset memiliki sumber dan ketentuan tersendiri, di luar lisensi kode ini.

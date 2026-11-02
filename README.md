# LEGO Price Predictor

Prediksi harga pasar set LEGO (Current_Price) dengan HistGradientBoosting,
lengkap dengan API FastAPI (deployment) dan monitoring drift/performa.

## Isi
- `Lego_set_and_prices_full.ipynb` - analisis, model, deployment, monitoring
- `deployment/` - API, modul fitur, modul monitoring, model tersimpan
- `sets.csv` - data

## Cara menjalankan
1. Taruh notebook dan `sets.csv` di folder yang sama.
2. Jalankan sel instalasi: `pip install fastapi uvicorn httpx pydantic joblib scipy`
3. Kernel -> Restart & Run All.

## Data
Dataset `sets.csv` diambil dari Kaggle:
- Judul dataset: LEGO Sets and Prices Over Time
- Pembuat: Alex Racape
- Link: https://www.kaggle.com/datasets/alexracape/lego-sets-and-prices-over-time
- Lisensi: MIT

---

# Kumpulan Notebook Klasifikasi

Folder `Klasifikasi Binner`, `Klasifikasi Multiclass`, dan `Klasifikasi Multilabel` berisi analisis dan pemodelan untuk tiga jenis tugas klasifikasi.

## Daftar Dataset

### A. Klasifikasi Biner (Binary Classification)

| No | Dataset | Sumber | Keterangan |
|----|---------|--------|------------|
| 1 | CIC-IDS2017 (Intrusion Detection System) | [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) | Hanya notebook yang disertakan. Data CSV (± 800 MB) perlu diunduh sendiri |
| 2 | Predicting Pulsar Star | [Kaggle](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate) | Data disertakan |
| 3 | Water Quality (Potability) | [Kaggle](https://www.kaggle.com/datasets/tamilamudan/water-quality-potability) | Data disertakan |

### B. Klasifikasi Multikelas (Multiclass Classification)

| No | Dataset | Sumber | Keterangan |
|----|---------|--------|------------|
| 1 | Sign Language MNIST | [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) | `sign_mnist_train.csv` tidak disertakan (terlalu besar), perlu diunduh sendiri |
| 2 | UNSW-NB15 Network Traffic | [Kaggle](https://www.kaggle.com/datasets/alextamboli/unsw-nb15) | File training/testing disertakan, file `UNSW-NB15_1..4.csv` tidak |
| 3 | Rice Type Classification | Kaggle (tautan menyusul) | Data disertakan |

### C. Klasifikasi Multilabel (Multilabel Classification)

| No | Dataset | Sumber | Keterangan |
|----|---------|--------|------------|
| 1 | Multi-label Prompt Dataset | [Kaggle](https://www.kaggle.com/datasets/nasimrajlaskar/multi-label-prompt-dataset) | Data disertakan |
| 2 | Genre Classification Dataset IMDb | [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) | Data disertakan |
| 3 | Multi-Label Classification Dataset | [Kaggle](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset) | Data disertakan |

## Cara Menjalankan

1. Clone repository ini.
2. Untuk dataset yang tidak disertakan, unduh dari tautan Kaggle di atas, lalu letakkan CSV-nya di folder notebook yang bersangkutan.
3. Pasang library: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
4. Jalankan `jupyter lab`, buka notebook, lalu **Run All** dari sel pertama.

## Catatan

Seluruh dataset berasal dari Kaggle dan digunakan sesuai lisensi masing-masing. Hak cipta data tetap milik pembuat aslinya.

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

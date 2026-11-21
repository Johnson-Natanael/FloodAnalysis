from flask import Flask, render_template, request
import os
import sys
import json
import pandas as pd
import numpy as np
from ee.ee_exception import EEException

# Menambahkan folder induk ke sys.path agar modul di folder 'modules/' dapat diimpor
sys.path.append("..")
from modules import flood_model

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


# Load data statis 16 kejadian banjir historis (2016-2024) dari file CSV lokal saat server dimulai
try:
    csv_path = os.path.join(os.path.dirname(__file__), "data_banjir_historis.csv")
    df_historis = pd.read_csv(csv_path)
    df_historis = df_historis.replace({np.nan: None})
    HISTORICAL_DATA = df_historis.to_dict(orient="records")
except Exception as e:
    print(f"Gagal memuat CSV: {e}")
    HISTORICAL_DATA = []

# Cache memori sementara untuk GeoJSON batas wilayah Bandung Raya agar tidak diunduh berulang dari GEE
BANDUNG_RAYA_GEOJSON_CACHE = None


def get_bandung_raya_admin_geojson():
    """
    Mengambil dan menyimpan cache batas administrasi kecamatan Bandung Raya.

    Data hanya ditarik satu kali dari GEE ketika cache masih kosong (None),
    lalu disimpan di memori server untuk pemanggilan berikutnya.
    """
    global BANDUNG_RAYA_GEOJSON_CACHE
    if BANDUNG_RAYA_GEOJSON_CACHE is None:
        try:
            BANDUNG_RAYA_GEOJSON_CACHE = flood_model.get_bandung_raya_geojson()
        except Exception as e:
            print(f"Gagal memuat batas administrasi Bandung Raya: {e}")
            BANDUNG_RAYA_GEOJSON_CACHE = {"type": "FeatureCollection", "features": []}
    return BANDUNG_RAYA_GEOJSON_CACHE


@app.route("/")
def dashboard():
    # Menampilkan halaman utama dashboard berisi ringkasan data dan grafik tren tahunan.
    # Mengirim data historis ke dashboard.html untuk dirender oleh Chart.js
    return render_template('dashboard.html', flood_data=HISTORICAL_DATA)


@app.route("/detail")
def detail():
    """
    Menampilkan halaman observasi historis berbasis peta interaktif Leaflet.js.

    Data poligon batas administrasi 84 kecamatan dimuat secara asinkron oleh JavaScript
    dari file GeoJSON statis agar halaman dapat langsung terbuka tanpa menunggu query GEE.
    """
    return render_template(
        "detail.html",
        flood_data=HISTORICAL_DATA
    )


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Menangani halaman analisis/prediksi area banjir baru secara dinamis melalui GEE.

    - GET  : Menampilkan form pemilihan kecamatan (84 kecamatan + Bandung Raya) dan rentang tanggal.
    - POST : Memvalidasi input, menjalankan model Sentinel-1 SAR di flood_model.py,
             lalu mengirim URL tile peta, thumbnail, luas genangan, dan GeoJSON ke halaman hasil.
    """
    if request.method == 'POST':
        kecamatan = request.form.get("kecamatan")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        print(f"Memproses Area: {kecamatan} | Tanggal: {start_date} hingga {end_date}")

        # Validasi kelengkapan seluruh kolom formulir sebelum memanggil server GEE
        if not start_date or not end_date or not kecamatan:
            return render_template("predict.html", error="Semua form harus diisi!", show_map=False)

        # Validasi urutan tanggal agar jendela waktu pasca-banjir bernilai valid
        if start_date > end_date:
            return render_template("predict.html", error="Tanggal mulai tidak boleh melebihi tanggal akhir.", show_map=False)

        try:
            # Menjalankan pipeline deteksi banjir Sentinel-1 SAR dan mengambil seluruh layer peta
            hasil = flood_model.get_flood_analysis(kecamatan, start_date, end_date)
            return render_template(
                "predict.html",
                kecamatan=kecamatan,
                start_date=start_date,
                end_date=end_date,
                luas_m2=round(hasil["luas_m2"], 2),
                luas_ha=round(hasil["luas_ha"], 2),
                pre_period=hasil["pre_period"],
                post_period=hasil["post_period"],
                # URL tile dinamis GEE (getMapId) untuk ditumpuk di peta interaktif Leaflet
                tile_sar_sebelum=hasil["tile_sar_sebelum"],
                tile_sar_sesudah=hasil["tile_sar_sesudah"],
                tile_sar_masked=hasil["tile_sar_masked"],
                tile_flood_mask=hasil["tile_flood_mask"],
                # [DINONAKTIFKAN] - Lapisan RGB Sentinel-2 dinonaktifkan karena sering tertutup awan (berubah dari rencana awal)
                # tile_rgb_basemap=hasil["tile_rgb_basemap"],
                # tile_rgb_masked=hasil["tile_rgb_masked"],
                # URL gambar PNG statis (getThumbUrl) untuk tampilan preview grid & tombol unduh
                thumb_sar_sebelum=hasil["thumb_sar_sebelum"],
                thumb_sar_sesudah=hasil["thumb_sar_sesudah"],
                thumb_sar_masked=hasil["thumb_sar_masked"],
                # [DINONAKTIFKAN] - Lapisan RGB Sentinel-2 dinonaktifkan karena sering tertutup awan
                # thumb_rgb_basemap=hasil["thumb_rgb_basemap"],
                # thumb_rgb_masked=hasil["thumb_rgb_masked"],
                thumb_flood_mask=hasil["thumb_flood_mask"],
                # Data GeoJSON diserialisasi ke string JSON agar bisa dibaca langsung oleh JavaScript
                roi_geojson=json.dumps(hasil["roi_geojson"]),
                flood_boundary_geojson=json.dumps(hasil["flood_boundary_geojson"]),
                bandung_raya_geojson=json.dumps(hasil["bandung_raya_geojson"]),
                # Titik tengah dan batas koordinat agar peta otomatis zoom tepat ke area fokus
                map_center=json.dumps(hasil["map_center"]),
                roi_bounds=json.dumps(hasil["roi_bounds"]),
                show_map=True
            )
        except EEException as ee_err:
            # Menangani gangguan khusus dari layanan komputasi Google Earth Engine (misal timeout/kuota)
            return render_template("predict.html", error=f"Kesalahan GEE: {str(ee_err)}", show_map=False)
        except Exception as e:
            # Menangani kesalahan umum di sisi server Python
            return render_template("predict.html", error=f"Terjadi kesalahan internal: {str(e)}", show_map=False)

    return render_template('predict.html', show_map=False)


if __name__ == "__main__":
    app.run(debug=True)
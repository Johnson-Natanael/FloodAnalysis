from flask import Flask, render_template, request
import sys
import pandas as pd
import numpy as np
from ee.ee_exception import EEException

sys.path.append("..")
from modules import flood_model

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# =========================================================================
# LOAD DATA STATIS HISTORIS DARI CSV
# =========================================================================
try:
    df_historis = pd.read_csv("data_banjir_historis.csv")
    df_historis = df_historis.replace({np.nan: None})
    HISTORICAL_DATA = df_historis.to_dict(orient="records")
except Exception as e:
    print(f"Gagal memuat CSV: {e}")
    HISTORICAL_DATA = []

@app.route("/")
def dashboard():
    # Mengirim data ke dashboard untuk visualisasi grafik agregasi
    return render_template('dashboard.html', flood_data=HISTORICAL_DATA)

@app.route("/detail")
def detail():
    return render_template("detail.html", flood_data=HISTORICAL_DATA)

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        kecamatan = request.form.get("kecamatan")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        print(f"Memproses Area: {kecamatan} | Tanggal: {start_date} hingga {end_date}")

        if not start_date or not end_date or not kecamatan:
            return render_template("predict.html", error="Semua form harus diisi!", show_img_container=False)
        
        if start_date > end_date:
            return render_template("predict.html", error="Tanggal mulai tidak boleh melebihi tanggal akhir.", show_img_container=False)
        
        try:
            hasil_banjir = flood_model.get_flood_analysis(kecamatan, start_date, end_date)
            return render_template(
                "predict.html",
                kecamatan=kecamatan,
                start_date=start_date,
                end_date=end_date,
                luas_m2=round(hasil_banjir["luas_m2"], 2),
                luas_ha=round(hasil_banjir["luas_ha"], 2),
                url_gambar=hasil_banjir["url_gambar"],
                pre_period=hasil_banjir["pre_period"],
                post_period=hasil_banjir["post_period"],
                show_img_container=True
            )
        except EEException as ee_err:
            return render_template("predict.html", error=f"Kesalahan GEE: {str(ee_err)}", show_img_container=False)
        except Exception as e:
            return render_template("predict.html", error=f"Terjadi kesalahan internal: {str(e)}", show_img_container=False)
    
    return render_template('predict.html', show_img_container=False)

if __name__ == "__main__":
    app.run(debug=True)
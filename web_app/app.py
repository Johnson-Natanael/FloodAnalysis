from flask import Flask, render_template, request, jsonify
import sys
from ee.ee_exception import EEException
sys.path.append("..")
from modules import sentinel_model
from modules import landsat_model
from modules import coastline
from modules import combine_hasil
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

df_perubahan = pd.read_csv("hasil_akresi_abrasi_per_tahun.csv")
coastlines_all, _ = combine_hasil.init_result()
@app.route("/")
def dashboard():
    return render_template('dashboard.html')

@app.route("/detail")
def detail():
    return render_template("detail.html", curYear=None, curStat=None)

@app.route("/detail/<year>/<status>")
def detail_with_params(year, status):

    # filter berdasarkan tahun
    data = df_perubahan[df_perubahan["tahun"] == int(year)]

    month_order = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "Mei": 5, "Jun": 6,
        "Jul": 7, "Agu": 8, "Sep": 9,
        "Okt": 10, "Nov": 11, "Des": 12
    }

    def get_month_value(period_text):
        # startdate contoh: "Jan_Mar"
        start_month = period_text.split("_")[0]
        return month_order.get(start_month, 0)

    # urutkan berdasarkan bulan pertama
    data = data.sort_values(
        by="startdate",
        key=lambda col: col.map(get_month_value)
    )

    # ubah ke list supaya mudah digunakan di HTML
    periodes = data.to_dict(orient="records")

    return render_template(
        "detail.html",
        curYear=year,
        curStat=status,
        periodes=periodes
    )

@app.route("/predict/<satelit>", methods=['GET', 'POST'])
def predict(satelit):
    # ambil tanggal dari form
    if request.method == 'POST':
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        print("Tanggal awal:", start_date)
        print("Tanggal akhir:", end_date)

        # cek tanggalnya diisi atau tidak
        if start_date == "" or end_date == "":
            return render_template(
                "predict.html",
                satelit=satelit,
                error="Tanggal mulai dan tanggal selesai harus diisi!",
                show_segment=False
            )
        
        # cek kalau masukkan tanggal salah
        if start_date > end_date:
            return render_template(
                "predict.html",
                satelit=satelit,
                error="Tanggal mulai tidak boleh melebihi tanggal akhir.",
                show_segment = False
            )
        
        filepath = f'../web_app/static/assets/custom_model/raw_data.tif'

        # catch exception kalau misalnya gaada data di rentang tanggal masukkan
        try:
            # jalankan model machine learning sesuai tipe satelit yang dipilih
            if satelit == 'sentinel':
                sentinel_model.init_predict_sentinel(start_date, end_date)
                coastline.extract_coastline_from_input(filepath, start_date, end_date)
            else:
                landsat_model.init_predict_landsat(start_date, end_date)
                coastline.extract_coastline_from_input(filepath, start_date, end_date)
        except EEException:
            return render_template(
                "predict.html",
                satelit=satelit,
                error="Data tidak ada pada tanggal yang ditentukan."
            )

        return render_template(
            "predict.html",
            satelit=satelit,
            start_date = start_date,
            end_date = end_date,
            show_img_container = "show"
        )
    
    return render_template('predict.html', satelit=satelit)

@app.route("/comparison", methods=['GET', 'POST'])
def comparison():
    if request.method == 'POST':
        # get tahun dari form
        start_year = request.form.get("start_yr")
        end_year = request.form.get("end_yr")
        print("Tahun awal:", start_year)
        print("Tahun akhir:", end_year)

        # cek tahun diisi atau tidak
        if not start_year or not end_year:
            return render_template(
                "comparison.html",
                error="Tahun awal dan tahun akhir harus diisi!",
                show_segment= "hide"
            )
        
        # cek kalau masukkan tahun salah
        if start_year > end_year:
            return render_template(
                "comparison.html",
                error="Tahun mulai tidak boleh melebihi tahun akhir.",
                show_segment = "hide"
            )
        
        # generate hasil perbandingan garis pantai
        # generate buat all sama rata-rata
        combine_hasil.generate_coastline_compare_new(int(start_year), int(end_year), coastlines_all)
        combine_hasil.generate_coastline_compare_average(int(start_year), int(end_year), coastlines_all)
        
        return render_template('comparison.html',
                               show_segment="show",
                               start_year=start_year,
                               end_year=end_year)
    return render_template('comparison.html')

if __name__ == "__main__":
    app.run(debug=True)
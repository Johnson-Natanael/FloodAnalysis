import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import ee

# Inisialisasi GEE
def init_ee():
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')

# Ambil batas administrasi ROI dari asset Bandung Raya
def get_asset_geometry(nama_area):
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/Bandung_Raya')
    
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
    
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)

    database_kecamatan = {
        "Pasteur": {"kc": "Sukajadi", "kk": "Kota Bandung"},
        "Cicaheum": {"kc": "Kiaracondong", "kk": "Kota Bandung"},
        "Sukajadi": {"kc": "Sukajadi", "kk": "Kota Bandung"},
        "Soreang": {"kc": "Soreang", "kk": "Bandung"},
        "Baleendah": {"kc": "Baleendah", "kk": "Bandung"},
        "Dayeuhkolot": {"kc": "Dayeuhkolot", "kk": "Bandung"},
        "Bojongsoang": {"kc": "Bojongsoang", "kk": "Bandung"},
        "Cilengkrang": {"kc": "Cilengkrang", "kk": "Bandung"},
        "Rancaekek": {"kc": "Rancaekek", "kk": "Bandung"},
        "Arjasari": {"kc": "Arjasari", "kk": "Bandung"},
        "Kutawaringin": {"kc": "Kutawaringin", "kk": "Bandung"},
        "Cileunyi": {"kc": "Cileunyi", "kk": "Bandung"},
        "Jatinangor": {"kc": "Jatinangor", "kk": "Sumedang"},
        "Sukasari": {"kc": "Sukasari", "kk": "Sumedang"},
        "Sukasari_KotaBandung": {"kc": "Sukasari", "kk": "Kota Bandung"},
    }
    
    if nama_area in database_kecamatan:
        target_kc = database_kecamatan[nama_area]['kc']
        target_kk = database_kecamatan[nama_area]['kk']
        roi_feature = wilayah_bandung_raya.filter(ee.Filter.And(
            ee.Filter.eq('WADMKC', target_kc),
            ee.Filter.eq('WADMKK', target_kk)
        ))
    else:
        roi_feature = wilayah_bandung_raya.filter(ee.Filter.eq('WADMKC', nama_area))
        
    return roi_feature.geometry()

# Perhitungan luas genangan m2 menggunakan model Backscatter Ratio
def calculate_flood_area_m2(roi, flood_start, flood_end):
    date_format = "%Y-%m-%d"
    f_start_date = datetime.datetime.strptime(flood_start, date_format)
    pre_start = (f_start_date - datetime.timedelta(days=30)).strftime(date_format)
    pre_end = (f_start_date - datetime.timedelta(days=1)).strftime(date_format)

    # 1. Panggil koleksi Sentinel-1 GRD IW
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW')))

    # 2. Ambil median sebelum dan sesudah
    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(flood_start, flood_end).median().clip(roi)

    # 3. Filter speckle 30m
    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    # 4. Hitung rasio backscatter band VH
    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))

    # 5. Terapkan threshold 1.30
    floodMask = floodRatio.gt(1.30)

    # 6. Masking lereng < 5 derajat dan piksel terhubung >= 10
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    # 7. Hitung total luas area
    floodArea = floodMask.multiply(ee.Image.pixelArea())
    totalFloodArea = floodArea.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )

    luas_m2 = totalFloodArea.getInfo().get('VH', 0)
    return luas_m2 if luas_m2 is not None else 0.0

def main():
    print("Mulai batch testing validasi musiman (kemarau vs penghujan)...")

    init_ee()
    os.makedirs("results", exist_ok=True)

    # Daftar kecamatan sampel rawan banjir di Bandung Raya
    kecamatan_list = [
        "Dayeuhkolot",
        "Baleendah",
        "Bojongsoang",
        "Rancaekek",
        "Kutawaringin",
        "Soreang"
    ]

    # Pasangan periode pengujian kemarau (Juli) dan penghujan (Nov-Jan)
    test_periods = [
        {
            "tahun": 2021,
            "kemarau": {"start": "2021-07-01", "end": "2021-07-15", "desc": "Kemarau Jul 2021"},
            "penghujan": {"start": "2021-11-02", "end": "2021-11-16", "desc": "Penghujan Nov 2021"}
        },
        {
            "tahun": 2022,
            "kemarau": {"start": "2022-07-01", "end": "2022-07-15", "desc": "Kemarau Jul 2022"},
            "penghujan": {"start": "2022-11-14", "end": "2022-11-28", "desc": "Penghujan Nov 2022"}
        },
        {
            "tahun": 2023,
            "kemarau": {"start": "2023-07-01", "end": "2023-07-15", "desc": "Kemarau Jul 2023"},
            "penghujan": {"start": "2023-12-25", "end": "2024-01-08", "desc": "Penghujan Des 2023"}
        },
        {
            "tahun": 2024,
            "kemarau": {"start": "2024-07-01", "end": "2024-07-15", "desc": "Kemarau Jul 2024"},
            "penghujan": {"start": "2024-01-11", "end": "2024-01-25", "desc": "Penghujan Jan 2024"}
        }
    ]

    records = []
    total_tests = len(kecamatan_list) * len(test_periods) * 2
    current_test = 0

    print(f"Total kombinasi yang akan diuji: {total_tests} (6 kecamatan x 4 tahun x 2 musim)")

    for period in test_periods:
        tahun = period["tahun"]
        for musim_key in ["kemarau", "penghujan"]:
            p_info = period[musim_key]
            m_label = "Kemarau" if musim_key == "kemarau" else "Penghujan"

            for kec in kecamatan_list:
                current_test += 1
                print(f"[{current_test}/{total_tests}] {kec} | {m_label} {tahun} ({p_info['start']} s/d {p_info['end']})...", end="", flush=True)

                try:
                    roi = get_asset_geometry(kec)
                    luas_m2 = calculate_flood_area_m2(roi, p_info['start'], p_info['end'])
                    luas_ha = luas_m2 / 10000.0
                    print(f" Selesai: {luas_m2:,.2f} m² ({luas_ha:.2f} ha)")
                except Exception as e:
                    print(f" Error: {e}")
                    luas_m2 = None
                    luas_ha = None

                records.append({
                    "Kecamatan": kec,
                    "Tahun": tahun,
                    "Musim": m_label,
                    "Periode_Mulai": p_info['start'],
                    "Periode_Selesai": p_info['end'],
                    "Deskripsi": p_info['desc'],
                    "Luas_m2": luas_m2,
                    "Luas_ha": luas_ha
                })
                # Beri jeda 1 detik agar tidak terkena batas kuota GEE
                time.sleep(1)

    df = pd.DataFrame(records)
    csv_path = os.path.join("results", "seasonal_validation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nHasil mentah berhasil disimpan di: {csv_path}")

    # Rekap perbandingan per tahun
    pivot_ha = df.pivot_table(index=["Kecamatan", "Tahun"], columns="Musim", values="Luas_ha").reset_index()
    pivot_ha["Rasio_Penghujan_vs_Kemarau"] = pivot_ha.apply(
        lambda r: r["Penghujan"] / r["Kemarau"] if (r["Kemarau"] > 0) else (float('inf') if r["Penghujan"] > 0 else 1.0),
        axis=1
    )

    rekap_csv = os.path.join("results", "seasonal_comparison_summary.csv")
    pivot_ha.to_csv(rekap_csv, index=False)

    # Rata-rata per kecamatan
    avg_per_kec = df.groupby(["Kecamatan", "Musim"])["Luas_ha"].mean().unstack()
    avg_per_kec["Rasio (P/K)"] = avg_per_kec["Penghujan"] / avg_per_kec["Kemarau"].replace(0, 0.001)

    print("\nRata-rata luas genangan terdeteksi (ha):")
    print(avg_per_kec.to_string())

    # Buat grafik perbandingan dengan matplotlib
    plt.figure(figsize=(12, 6))
    x = range(len(avg_per_kec.index))
    width = 0.35

    plt.bar([i - width/2 for i in x], avg_per_kec["Kemarau"], width=width, label="Musim Kemarau (Juli)", color="#D97706")
    plt.bar([i + width/2 for i in x], avg_per_kec["Penghujan"], width=width, label="Musim Penghujan (Nov-Jan)", color="#2563EB")

    plt.xlabel("Kecamatan", fontsize=12, fontweight="bold")
    plt.ylabel("Rata-rata Luas Genangan Terdeteksi (Hektar)", fontsize=12, fontweight="bold")
    plt.title("Validasi Musiman Model Deteksi Banjir: Kemarau vs Penghujan di Bandung Raya", fontsize=14, fontweight="bold")
    plt.xticks(x, avg_per_kec.index, rotation=15, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_path = os.path.join("results", "seasonal_comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Grafik perbandingan disimpan di: {chart_path}")

if __name__ == "__main__":
    main()

import ee
import requests
import os
import time

def init_ee():
    """Inisialisasi koneksi ke Google Earth Engine."""
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')

def get_manual_geometry(kecamatan_name):
    """Menyimpan dan memanggil poligon area administrasi."""
    polygons = {
        "BandungRaya": ee.Geometry.Polygon([[[107.4877048585857, -6.801766035473307], [107.4877048585857, -7.08531528322579], [107.86810646991383, -7.08531528322579], [107.86810646991383, -6.801766035473307]]]),
        "Arjasari": ee.Geometry.Polygon([[[107.67212057384126, -7.069800280233803], [107.67212057384126, -7.0802770693175106], [107.68799925120942, -7.0802770693175106], [107.68799925120942, -7.069800280233803]]]),
        "Baleendah": ee.Geometry.Polygon([[[107.62063785527936, -6.9985378362462205], [107.62063785527936, -7.01455350430119], [107.64484210942975, -7.01455350430119], [107.64484210942975, -6.9985378362462205]]]),
        "Bojongsoang": ee.Geometry.Polygon([[[107.63536870211537, -6.968848875045322], [107.63536870211537, -6.99696290680392], [107.67390668124135, -6.99696290680392], [107.67390668124135, -6.968848875045322]]]),
        "Cicaheum": ee.Geometry.Polygon([[[107.64643943246303, -6.901737206196583], [107.64643943246303, -6.913794088336616], [107.66214644845424, -6.913794088336616], [107.66214644845424, -6.901737206196583]]]),
        "Cilengkrang": ee.Geometry.Polygon([[[107.68684537471263, -6.898863941823968], [107.68684537471263, -6.911389535902242], [107.7041831737849, -6.911389535902242], [107.7041831737849, -6.898863941823968]]]),
        "Cileunyi": ee.Geometry.Polygon([[[107.72993665564016, -6.929554795799393], [107.72993665564016, -6.946084026670311], [107.75328260290578, -6.946084026670311], [107.75328260290578, -6.929554795799393]]]),
        "Dayeuhkolot": ee.Geometry.Polygon([[[107.59065091341908, -6.964333433818634], [107.59065091341908, -6.9917661979804295], [107.63142049044545, -6.9917661979804295], [107.63142049044545, -6.964333433818634]]]),
        "Kutawaringin": ee.Geometry.Polygon([[[107.51818801011933, -7.0017135208836025], [107.51818801011933, -7.012532602412637], [107.53252173509492, -7.012532602412637], [107.53252173509492, -7.0017135208836025]]]),
        "Pasteur": ee.Geometry.Polygon([[[107.58014267405795, -6.886767749107755], [107.58014267405795, -6.896737381051292], [107.59321039637851, -6.896737381051292], [107.59321039637851, -6.886767749107755]]]),
        "Rancaekek": ee.Geometry.Polygon([[[107.79221372162944, -6.972790975332603], [107.79221372162944, -6.991107730553906], [107.81564549958354, -6.991107730553906], [107.81564549958354, -6.972790975332603]]]),
        "Soreang": ee.Geometry.Polygon([[[107.50979096018011, -7.017948839640189], [107.50979096018011, -7.029236064809415], [107.52549797617132, -7.029236064809415], [107.52549797617132, -7.017948839640189]]])
    }
    # Jika nama daerah tidak ada, default menggunakan batas maksimal BandungRaya
    return polygons.get(kecamatan_name, polygons.get("BandungRaya"))

def get_flood_analysis(kecamatan_name, pre_start, pre_end, post_start, post_end):
    """
    Fungsi untuk mendeteksi banjir Sentinel-1 dengan input periode yang sudah fix.
    """
    # 1. Tentukan ROI
    roi = get_manual_geometry(kecamatan_name)
    
    # Ekstrak tahun dari variabel post_start untuk penamaan file
    tahun = post_start.split('-')[0]
    
    # 2. Panggil koleksi Sentinel-1 GRD
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))

    # 3. Ambil rata-rata (median) berdasar periode manual
    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(post_start, post_end).median().clip(roi)

    # 4. Speckle Filtering
    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    # 5. Backscatter Ratio (BR) & Thresholding
    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))
    floodMask = floodRatio.gt(1.30)

    # 6. Pemurnian Hasil
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    # 7. Hitung Luas Area Banjir
    floodArea = floodMask.multiply(ee.Image.pixelArea())
    totalFloodArea = floodArea.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    
    luas_m2 = totalFloodArea.getInfo().get('VH', 0)
    if luas_m2 is None:
        luas_m2 = 0

    # # =========================================================================
    # # 8. PROSES VISUALISASI DAN DOWNLOAD (DENGAN SISTEM RETRY & TIMEOUT LAMA)
    # # =========================================================================
    # visVHSebelum = s1FilteredSebelum.select('VH').clip(roi).visualize(
    #     min=-25, max=0, palette=['black', 'white']
    # )
    # visVHSesudah = s1FilteredSesudah.select('VH').clip(roi).visualize(
    #     min=-25, max=0, palette=['black', 'white']
    # )
    # visFloodMask = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(
    #     palette=['blue']
    # )
    # overlayBanjir = visVHSesudah.blend(visFloodMask)

    # paramsUnduh = {'dimensions': 1000, 'region': roi, 'format': 'png'}

    # daftar_gambar = {
    #     "sebelum": visVHSebelum,
    #     "sesudah": visVHSesudah,
    #     "masked": overlayBanjir
    # }

    # file_terunduh = {}
    # # nama_folder = "imageBerita"
    # nama_folder = "imageBandungRaya"
    # os.makedirs(nama_folder, exist_ok=True)

    # for tipe, citra_obj in daftar_gambar.items():
    #     daerah_safe = kecamatan_name.replace(" ", "")
    #     nama_file = os.path.join(nama_folder, f"{tahun}_{tipe}_{daerah_safe}.png")
        
    #     # Dapatkan URL dari GEE
    #     try:
    #         url = citra_obj.getThumbURL(paramsUnduh)
    #     except Exception as e:
    #         print(f"    -> Gagal membuat URL untuk {tipe}. Error GEE: {e}")
    #         continue

    #     # --- SISTEM RETRY UNTUK MENGATASI TIMEOUT ---
    #     maksimal_percobaan = 3
    #     berhasil_unduh = False
        
    #     for percobaan in range(1, maksimal_percobaan + 1):
    #         try:
    #             # Perbesar timeout menjadi 120 detik (2 menit)
    #             respons = requests.get(url, timeout=120) 
                
    #             if respons.status_code == 200:
    #                 with open(nama_file, 'wb') as f:
    #                     f.write(respons.content)
    #                 print(f"    -> Tersimpan: {nama_file}")
    #                 file_terunduh[tipe] = os.path.abspath(nama_file)
    #                 berhasil_unduh = True
    #                 break # Jika sukses, keluar dari loop retry
    #             else:
    #                 print(f"    -> Gagal mengunduh {nama_file} (HTTP {respons.status_code}). Coba lagi...")
                    
    #         except requests.exceptions.Timeout:
    #             print(f"    -> Timeout pada percobaan {percobaan} untuk {nama_file}. Komputasi GEE butuh waktu lebih lama...")
    #         except requests.exceptions.RequestException as e:
    #             print(f"    -> Masalah koneksi pada percobaan {percobaan}: {e}")
            
    #         # Berikan jeda 5 detik sebelum mencoba mengunduh lagi
    #         if percobaan < maksimal_percobaan:
    #             time.sleep(5) 
                
    #     if not berhasil_unduh:
    #          print(f"    -> [GAGAL] Telah mencoba {maksimal_percobaan} kali, namun tetap gagal mengunduh {nama_file}.")

    # return luas_m2 / 10000
    return luas_m2

# =========================================================================
# EKSEKUSI LOOPING BERDASARKAN DICTIONARY
# =========================================================================
if __name__ == "__main__":
    init_ee()
    
    # Dictionary dari seluruh riwayat kejadian banjir
    # Catatan: "Jalan Raya Dayeuhkolot-Banjaran" dialihkan ke region "Dayeuhkolot" agar ROI dikenali
    daftar_kejadian = [
        # --- TAHUN 2016 ---
        {"daerah": "Baleendah", "pre_start": "2016-03-12", "pre_end": "2016-04-11", "post_start": "2016-04-12", "post_end": "2016-04-26"},
        # --- TAHUN 2017 ---
        {"daerah": "Pasteur", "pre_start": "2017-02-08", "pre_end": "2017-03-07", "post_start": "2017-03-08", "post_end": "2017-03-22"},
        {"daerah": "Baleendah", "pre_start": "2017-02-08", "pre_end": "2017-03-07", "post_start": "2017-03-08", "post_end": "2017-03-22"},
        # --- TAHUN 2018 ---
        {"daerah": "Dayeuhkolot", "pre_start": "2018-10-12", "pre_end": "2018-11-11", "post_start": "2018-11-12", "post_end": "2018-11-26"},
        {"daerah": "Cicaheum", "pre_start": "2018-02-20", "pre_end": "2018-03-19", "post_start": "2018-03-20", "post_end": "2018-04-03"},
        # --- TAHUN 2019 ---
        {"daerah": "Cilengkrang", "pre_start": "2019-01-09", "pre_end": "2019-02-08", "post_start": "2019-02-09", "post_end": "2019-02-28"},
        {"daerah": "Bojongsoang", "pre_start": "2019-11-17", "pre_end": "2019-12-16", "post_start": "2019-12-17", "post_end": "2020-01-06"},
        # --- TAHUN 2020 ---
        {"daerah": "Rancaekek", "pre_start": "2020-11-15", "pre_end": "2020-12-14", "post_start": "2020-12-15", "post_end": "2020-12-29"},
        {"daerah": "Pasteur", "pre_start": "2020-11-25", "pre_end": "2020-12-24", "post_start": "2020-12-25", "post_end": "2021-01-08"},
        # --- TAHUN 2021 ---
        {"daerah": "Arjasari", "pre_start": "2021-09-19", "pre_end": "2021-10-18", "post_start": "2021-10-19", "post_end": "2021-11-02"},
        {"daerah": "Dayeuhkolot", "pre_start": "2021-10-02", "pre_end": "2021-11-01", "post_start": "2021-11-02", "post_end": "2021-11-16"},
        # --- TAHUN 2022 ---
        {"daerah": "Kutawaringin", "pre_start": "2022-10-14", "pre_end": "2022-11-13", "post_start": "2022-11-14", "post_end": "2022-11-28"},
        {"daerah": "Cileunyi", "pre_start": "2022-10-07", "pre_end": "2022-11-06", "post_start": "2022-11-07", "post_end": "2022-11-21"},
        # --- TAHUN 2023 ---
        {"daerah": "Soreang", "pre_start": "2023-10-30", "pre_end": "2023-11-29", "post_start": "2023-11-30", "post_end": "2023-12-14"},
        {"daerah": "Dayeuhkolot", "pre_start": "2023-11-25", "pre_end": "2023-12-24", "post_start": "2023-12-25", "post_end": "2024-01-08"},
        # --- TAHUN 2024 ---
        {"daerah": "Dayeuhkolot", "pre_start": "2023-12-11", "pre_end": "2024-01-10", "post_start": "2024-01-11", "post_end": "2024-01-25"},

        # # BandungRaya
        # # --- TAHUN 2016 ---
        # {"daerah": "BandungRaya1", "pre_start": "2016-03-12", "pre_end": "2016-04-11", "post_start": "2016-04-12", "post_end": "2016-04-26"},
        # # --- TAHUN 2017 ---
        # {"daerah": "BandungRaya1", "pre_start": "2017-02-08", "pre_end": "2017-03-07", "post_start": "2017-03-08", "post_end": "2017-03-22"},
        # {"daerah": "BandungRaya2", "pre_start": "2017-02-08", "pre_end": "2017-03-07", "post_start": "2017-03-08", "post_end": "2017-03-22"},
        # # --- TAHUN 2018 ---
        # {"daerah": "BandungRaya1", "pre_start": "2018-10-12", "pre_end": "2018-11-11", "post_start": "2018-11-12", "post_end": "2018-11-26"},
        # {"daerah": "BandungRaya2", "pre_start": "2018-02-20", "pre_end": "2018-03-19", "post_start": "2018-03-20", "post_end": "2018-04-03"},
        # # --- TAHUN 2019 ---
        # {"daerah": "BandungRaya1", "pre_start": "2019-01-09", "pre_end": "2019-02-08", "post_start": "2019-02-09", "post_end": "2019-02-28"},
        # {"daerah": "BandungRaya2", "pre_start": "2019-11-17", "pre_end": "2019-12-16", "post_start": "2019-12-17", "post_end": "2020-01-06"},
        # # --- TAHUN 2020 ---
        # {"daerah": "BandungRaya1", "pre_start": "2020-11-15", "pre_end": "2020-12-14", "post_start": "2020-12-15", "post_end": "2020-12-29"},
        # {"daerah": "BandungRaya2", "pre_start": "2020-11-25", "pre_end": "2020-12-24", "post_start": "2020-12-25", "post_end": "2021-01-08"},
        # # --- TAHUN 2021 ---
        # {"daerah": "BandungRaya1", "pre_start": "2021-09-19", "pre_end": "2021-10-18", "post_start": "2021-10-19", "post_end": "2021-11-02"},
        # {"daerah": "BandungRaya2", "pre_start": "2021-10-02", "pre_end": "2021-11-01", "post_start": "2021-11-02", "post_end": "2021-11-16"},
        # # --- TAHUN 2022 ---
        # {"daerah": "BandungRaya1", "pre_start": "2022-10-14", "pre_end": "2022-11-13", "post_start": "2022-11-14", "post_end": "2022-11-28"},
        # {"daerah": "BandungRaya2", "pre_start": "2022-10-07", "pre_end": "2022-11-06", "post_start": "2022-11-07", "post_end": "2022-11-21"},
        # # --- TAHUN 2023 ---
        # {"daerah": "BandungRaya1", "pre_start": "2023-10-30", "pre_end": "2023-11-29", "post_start": "2023-11-30", "post_end": "2023-12-14"},
        # {"daerah": "BandungRaya2", "pre_start": "2023-11-25", "pre_end": "2023-12-24", "post_start": "2023-12-25", "post_end": "2024-01-08"},
        # # --- TAHUN 2024 ---
        # {"daerah": "BandungRaya1", "pre_start": "2023-12-11", "pre_end": "2024-01-10", "post_start": "2024-01-11", "post_end": "2024-01-25"},
    ]

    print(f"Memulai proses analisis untuk {len(daftar_kejadian)} kejadian...\n")
    
    for i, event in enumerate(daftar_kejadian, 1):
        daerah = event['daerah']
        tahun = event['post_start'].split('-')[0]
        
        print(f"[{i}/{len(daftar_kejadian)}] Menganalisis {daerah} (Tahun {tahun})")
        print(f"    Periode Sebelum: {event['pre_start']} s/d {event['pre_end']}")
        print(f"    Periode Sesudah: {event['post_start']} s/d {event['post_end']}")
        
        # Eksekusi fungsi utama
        # luas_ha = get_flood_analysis(
        #     kecamatan_name=daerah,
        #     pre_start=event['pre_start'],
        #     pre_end=event['pre_end'],
        #     post_start=event['post_start'],
        #     post_end=event['post_end']
        # )
        luas_m2 = get_flood_analysis(
            kecamatan_name=daerah,
            pre_start=event['pre_start'],
            pre_end=event['pre_end'],
            post_start=event['post_start'],
            post_end=event['post_end']
        )
        
        # print(f"    Selesai. Luas terdampak: {luas_ha:.2f} Ha\n")
        print(f"    Selesai. Luas terdampak: {luas_m2:.2f} m²\n")
        
        # Jeda 2 detik agar tidak terkena limit API Earth Engine secara tiba-tiba
        time.sleep(2)
        
    print("Semua proses selesai dieksekusi!")
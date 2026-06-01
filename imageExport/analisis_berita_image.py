import ee
import requests
import os
import time

def init_ee():
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')

def get_asset_geometry(nama_area):
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/Bandung_Raya')
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
    
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)

    if nama_area.startswith("BandungRaya"):
        return wilayah_bandung_raya.geometry().dissolve()

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
        "Cimanggung": {"kc": "Cimanggung", "kk": "Sumedang"},
        "Tanjungsari": {"kc": "Tanjungsari", "kk": "Sumedang"},
        "Pamulihan": {"kc": "Pamulihan", "kk": "Sumedang"},
        "Sukasari": {"kc": "Sukasari", "kk": "Sumedang"} 
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

def get_flood_analysis(kecamatan_name, pre_start, pre_end, post_start, post_end):
    roi = get_asset_geometry(kecamatan_name)
    tahun = post_start.split('-')[0]
    
    # Reload aset untuk rekonstruksi Peta Konteks seluruh Bandung Raya
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/Bandung_Raya')
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)
    
    # Sentinel-1 Processing
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))

    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(post_start, post_end).median().clip(roi)

    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))
    floodMask = floodRatio.gt(1.30)

    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    floodArea = floodMask.multiply(ee.Image.pixelArea())
    totalFloodArea = floodArea.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e9)
    luas_m2 = totalFloodArea.getInfo().get('VH', 0) or 0

    # Sentinel-2 Basemap Statis (Opsi 1)
    s2Collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi)
                    .filterDate('2023-06-01', '2023-09-30')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    def maskS2clouds(image):
        qa = image.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask)
        
    s2_basemap = s2Collection.map(maskS2clouds).median().clip(roi)

    # Visualisasi Komponen Gambar
    visFloodMask = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(palette=['blue'])
    
    visVHSebelum = s1FilteredSebelum.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    visVHSesudah = s1FilteredSesudah.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    overlaySAR = visVHSesudah.blend(visFloodMask)
    
    visRGBPolos = s2_basemap.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    overlayRGB = visRGBPolos.blend(visFloodMask)
    
    kanvas_putih = ee.Image.constant(1).visualize(palette=['#FFFFFF']).clip(wilayah_bandung_raya.geometry().buffer(5000))
    outline_bandung_raya = wilayah_bandung_raya.style(color='#333333', fillColor='#00000000', width=2)
    roi_feature_coll = ee.FeatureCollection([ee.Feature(roi)])
    roi_merah = roi_feature_coll.style(color='#FF0000', fillColor='#FF0000', width=2)
    peta_konteks = kanvas_putih.blend(outline_bandung_raya).blend(roi_merah)

    # Mapping 7 Jenis Gambar yang akan Didownload ke Folder Lokal
    daftar_gambar = {
        "sar_sebelum": visVHSebelum,
        "sar_sesudah": visVHSesudah,
        "sar_masked": overlaySAR,
        "rgb_sebelum": visRGBPolos,
        "rgb_sesudah": visRGBPolos,
        "rgb_masked": overlayRGB,
        "peta_konteks": peta_konteks
    }

    # nama_folder = "imageBandungRaya"
    nama_folder = "imageBerita"
    os.makedirs(nama_folder, exist_ok=True)

    for tipe, citra_obj in daftar_gambar.items():
        daerah_safe = kecamatan_name.replace(" ", "")
        nama_file = os.path.join(nama_folder, f"{tahun}_{tipe}_{daerah_safe}.png")
        
        # Konfigurasi pembatas pengunduhan dinamis
        if tipe == "peta_konteks":
            paramsUnduh = {'dimensions': 1000, 'region': wilayah_bandung_raya.geometry().bounds(), 'format': 'png'}
        else:
            paramsUnduh = {'dimensions': 1000, 'region': roi.bounds(), 'format': 'png'}
        
        try:
            url = citra_obj.getThumbURL(paramsUnduh)
        except Exception as e:
            print(f"    -> Gagal membuat URL untuk {tipe}: {e}")
            continue

        maksimal_percobaan = 3
        berhasil_unduh = False
        
        for percobaan in range(1, maksimal_percobaan + 1):
            try:
                respons = requests.get(url, timeout=120) 
                if respons.status_code == 200:
                    with open(nama_file, 'wb') as f:
                        f.write(respons.content)
                    print(f"    -> Tersimpan: {nama_file}")
                    berhasil_unduh = True
                    break
                else:
                    print(f"    -> HTTP Error {respons.status_code} pada {tipe}. Mengulangi...")
            except requests.exceptions.Timeout:
                print(f"    -> Timeout percobaan {percobaan} untuk {tipe}...")
            except Exception as e:
                print(f"    -> Koneksi terganggu: {e}")
            
            if percobaan < maksimal_percobaan:
                time.sleep(5) 
                
        if not berhasil_unduh:
             print(f"    -> [GAGAL] Gagal mengunduh {nama_file} setelah {maksimal_percobaan} percobaan.")

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
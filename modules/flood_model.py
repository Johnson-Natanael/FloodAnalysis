import ee
import datetime

def init_ee():
    """Inisialisasi koneksi ke Google Earth Engine."""
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')

def get_asset_geometry(nama_area):
    """
    Memanggil poligon area administrasi asli dari Asset GEE.
    Mencakup 4 Wilayah Utama + 5 Kecamatan di Sumedang Barat.
    """
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


def get_flood_analysis(kecamatan_name, flood_start, flood_end):
    """
    Fungsi utama untuk mendeteksi banjir menggunakan Sentinel-1 
    dan menyandingkannya dengan citra optik statis Sentinel-2 (Opsi 1).
    """
    init_ee()

    # 1. Ambil geometri ROI dan keseluruhan wilayah Bandung Raya untuk peta konteks
    roi = get_asset_geometry(kecamatan_name)
    
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/BatasWilayahKecamatan')
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)
    
    # 2. Pengaturan Waktu Otomatis Sentinel-1
    date_format = "%Y-%m-%d"
    f_start_date = datetime.datetime.strptime(flood_start, date_format)
    pre_start = (f_start_date - datetime.timedelta(days=30)).strftime(date_format)
    pre_end = (f_start_date - datetime.timedelta(days=1)).strftime(date_format)
    
    # 3. Panggil koleksi data Sentinel-1 GRD
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW')))

    # 4. Ambil rata-rata (median)
    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(flood_start, flood_end).median().clip(roi)

    # 5. Speckle Filtering Sederhana
    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    # 6. Perhitungan Backscatter Ratio (BR) pada band VH
    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))

    # 7. Penerapan Ambang Batas (Thresholding = 1.30)
    floodMask = floodRatio.gt(1.30)

    # 8. Pemurnian Hasil (Slope Masking & Connected Pixels)
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    # 9. Hitung Luas Area Banjir (m^2)
    floodArea = floodMask.multiply(ee.Image.pixelArea())
    totalFloodArea = floodArea.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    
    luas_m2 = totalFloodArea.getInfo().get('VH', 0)
    if luas_m2 is None: luas_m2 = 0

    # =========================================================================
    # 10. EKSTRAKSI BASEMAP OPTIK SENTINEL-2 BEBAS AWAN (KEMARAU 2023)
    # =========================================================================
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

    # =========================================================================
    # 11. GENERASI VISUALISASI 7 GAMBAR
    # =========================================================================
    # A. Masker Banjir Biru Murni
    visFloodMask = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(palette=['blue'])

    # Gambar 1 & 2: SAR Grayscale (Sebelum & Sesudah)
    img_sar_sebelum = s1FilteredSebelum.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    img_sar_sesudah = s1FilteredSesudah.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    
    # Gambar 3: SAR Masked (Overlay di atas SAR Pasca)
    img_sar_masked = img_sar_sesudah.blend(visFloodMask)
    
    # Gambar 4 & 5: Optik RGB Polos (S2 Cloud-Free Basemap)
    img_rgb_polos = s2_basemap.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    
    # Gambar 6: Optik RGB Masked
    img_rgb_masked = img_rgb_polos.blend(visFloodMask)
    
    # Gambar 7: Peta Konteks Bandung Raya (Kanvas Putih + Batas Hitam + ROI Merah)
    kanvas_putih = ee.Image.constant(1).visualize(palette=['#FFFFFF']).clip(wilayah_bandung_raya.geometry().buffer(5000))
    outline_bandung_raya = wilayah_bandung_raya.style(color='#333333', fillColor='#00000000', width=2)
    roi_feature_coll = ee.FeatureCollection([ee.Feature(roi)])
    roi_merah = roi_feature_coll.style(color='#FF0000', fillColor='#FF0000', width=2)
    img_peta_konteks = kanvas_putih.blend(outline_bandung_raya).blend(roi_merah)

    # =========================================================================
    # 12. PARAMETER EKSPOR DAN GENERATE URL
    # =========================================================================
    paramsROI = {'dimensions': 1000, 'region': roi.bounds(), 'format': 'png'}
    paramsKonteks = {'dimensions': 1000, 'region': wilayah_bandung_raya.geometry().bounds(), 'format': 'png'}

    def safe_get_url(ee_img, params):
        try: return ee_img.getThumbURL(params)
        except Exception: return ""

    return {
        "luas_m2": luas_m2,
        "luas_ha": luas_m2 / 10000,
        "pre_period": f"{pre_start} s/d {pre_end}",
        "post_period": f"{flood_start} s/d {flood_end}",
        # 7 URL yang diminta:
        "url_sar_sebelum": safe_get_url(img_sar_sebelum, paramsROI),
        "url_sar_sesudah": safe_get_url(img_sar_sesudah, paramsROI),
        "url_sar_masked": safe_get_url(img_sar_masked, paramsROI),
        "url_rgb_sebelum": safe_get_url(img_rgb_polos, paramsROI),
        "url_rgb_sesudah": safe_get_url(img_rgb_polos, paramsROI),
        "url_rgb_masked": safe_get_url(img_rgb_masked, paramsROI),
        "url_peta_konteks": safe_get_url(img_peta_konteks, paramsKonteks),
        # Fallback untuk stabilitas app.py lama
        "url_gambar": safe_get_url(img_sar_masked, paramsROI)
    }
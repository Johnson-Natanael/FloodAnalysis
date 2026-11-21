import ee
import datetime


def init_ee():
    """
    Menginisialisasi koneksi ke Google Earth Engine menggunakan project 'ee-johnsonnn'.

    Jika sesi kredensial lokal belum aktif atau sudah kedaluwarsa, fungsi akan memicu
    ee.Authenticate() agar pengguna dapat login terlebih dahulu.
    """
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')


def get_asset_geometry(nama_area):
    """
    Mengambil poligon batas wilayah (ROI) dari asset GEE Bandung_Raya.

    Cakupan Bandung Raya dibentuk dari 4 wilayah utama (Kab. Bandung, Kab. Bandung Barat,
    Kota Bandung, Kota Cimahi) ditambah 5 kecamatan di Sumedang bagian barat (Cimanggung,
    Tanjungsari, Sukasari, Jatinangor, Pamulihan) total 84 kecamatan.

    Parameter:
        nama_area (str): Nama kecamatan, alias lokasi berita, atau awalan 'BandungRaya'.
    """
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/Bandung_Raya')

    # Memfilter fitur kabupaten/kota inti di Cekungan Bandung Raya
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))

    # Memfilter 5 kecamatan di Kabupaten Sumedang bagian barat yang masuk Cekungan Bandung
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))

    # Menggabungkan kedua FeatureCollection menjadi satu cakupan wilayah Bandung Raya
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)

    # Jika pengguna memilih skala makro regional, seluruh poligon kecamatan dilebur menjadi satu batas luar
    if nama_area.startswith("BandungRaya"):
        return wilayah_bandung_raya.geometry().dissolve()

    # Pemetaan alias lokasi populer (seperti Pasteur/Cicaheum) ke kecamatan resmi, sekaligus
    # memisahkan kecamatan bernama sama ('Sukasari' di Sumedang vs Kota Bandung)
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
        # Untuk nama kecamatan lainnya di luar daftar alias, langsung saring kolom WADMKC
        roi_feature = wilayah_bandung_raya.filter(ee.Filter.eq('WADMKC', nama_area))

    return roi_feature.geometry()


# def get_bandung_raya_geojson():
#     """
#     Mengambil data GeoJSON batas seluruh kecamatan di Bandung Raya beserta atribut WADMKC.

#     Digunakan sebagai lapisan batas administratif (overlay) agar pengguna mudah
#     mengenali posisi tiap kecamatan saat meninjau peta Leaflet.
#     """
#     init_ee()
#     batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/BatasWilayahKecamatan')
#     daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
#     wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
#     daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
#     sumedang_barat = (batas_wilayah
#                       .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
#                       .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
#     wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)
#     return wilayah_bandung_raya.getInfo()


def get_flood_analysis(kecamatan_name, flood_start, flood_end):
    """
    Menjalankan alur lengkap deteksi genangan banjir menggunakan Sentinel-1 SAR di GEE.

    Tahapan utama:
    1) Mengambil poligon batas ROI kecamatan/Bandung Raya.
    2) Membentuk jendela waktu acuan 30 hari sebelum tanggal mulai banjir.
    3) Mengambil citra Sentinel-1 GRD mode IW polarisasi VV/VH yang mampu menembus awan.
    4) Menggabungkan citra dengan komposit median lalu menerapkan filter focal_mean radius 30 m.
    5) Menghitung Backscatter Ratio (VH_sesudah / VH_sebelum > 1.30 pada nilai dB negatif).
    6) Membersihkan false positive lereng curam (kemiringan >= 5 derajat) dan noise (< 10 piksel).
    7) Menghitung total luas piksel banjir (resolusi 10 m) serta menyiapkan tile URL dan GeoJSON.
    """
    init_ee()

    # 1. Ambil geometri ROI target dan FeatureCollection Bandung Raya untuk layer konteks peta
    roi = get_asset_geometry(kecamatan_name)

    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/BatasWilayahKecamatan')
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))
    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)

    # 2. Pengaturan rentang tanggal pra-banjir otomatis (30 hari tepat sebelum kejadian)
    # Rentang 30 hari menjaga ketersediaan minimal beberapa sapuan Sentinel-1 tanpa bergeser musim
    date_format = "%Y-%m-%d"
    f_start_date = datetime.datetime.strptime(flood_start, date_format)
    pre_start = (f_start_date - datetime.timedelta(days=30)).strftime(date_format)
    pre_end = (f_start_date - datetime.timedelta(days=1)).strftime(date_format)

    # 3. Panggil koleksi citra radar Sentinel-1 GRD (nilai piksel sudah dalam satuan desibel / dB negatif)
    # Dipilih mode IW (Interferometric Wide Swath) beresolusi 10 m dengan polarisasi ganda VV dan VH
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW')))

    # 4. Membuat komposit median pra-banjir dan pasca-banjir untuk meredam variasi sesaat antar-sapuan
    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(flood_start, flood_end).median().clip(roi)

    # 5. Speckle Filtering memakai focal_mean lingkaran radius 30 meter
    # Langkah ini menghaluskan bintik noise radar (speckle) sebelum nilai piksel dibandingkan
    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    # 6. Perhitungan Backscatter Ratio (BR) pada polarisasi VH
    # Catatan penting skala dB: permukaan air tenang memantulkan gelombang radar menjauh (specular reflection)
    # sehingga nilai dB saat banjir menjadi jauh lebih negatif dibanding pra-banjir (misal -21 dB vs -14 dB).
    # Akibatnya, pembagian nilai dB negatif sesudah / sebelum menghasilkan rasio lebih besar dari 1 (> 1.30).
    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))

    # 7. Penerapan ambang batas (thresholding > 1.30) untuk membentuk masker biner genangan awal
    floodMask = floodRatio.gt(1.30)

    # 8. Pemurnian hasil deteksi (Slope Masking & Connected Pixel Count)
    # a) Masker kemiringan lereng < 5 derajat memakai DEM SRTM agar bayangan bukit/lereng tidak dikira banjir
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    # b) Menghitung ukuran klaster piksel berdekatan (maksimum pencarian 100) lalu membuang klaster kecil (< 10 piksel)
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    # 9. Menghitung total luas genangan (m^2) dengan mengalikan setiap piksel banjir terhadap luas riil pikselnya
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

    # 10. [DINONAKTIFKAN] - Basemap optik Sentinel-2 tidak dipakai lagi karena tutupan awan tinggi (70,36%)
    # s2Collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    #                 .filterBounds(roi)
    #                 .filterDate('2023-06-01', '2023-09-30')
    #                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    # 
    # def maskS2clouds(image):
    #     qa = image.select('QA60')
    #     cloudBitMask = 1 << 10
    #     cirrusBitMask = 1 << 11
    #     mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    #     return image.updateMask(mask)
    #     
    # s2_basemap = s2Collection.map(maskS2clouds).median().clip(roi)

    # 11. Generate GEE Tile URLs (getMapId) agar peta interaktif Leaflet bisa memuat layer satelit secara dinamis
    def safe_get_tile_url(ee_image):
        """Mengambil URL tile peta dari objek ee.Image dengan penanganan jika terjadi error."""
        try:
            map_id_dict = ee_image.getMapId()
            return map_id_dict['tile_fetcher'].url_format
        except Exception:
            return ""

    # Layer SAR Grayscale sebelum & sesudah banjir (skala visual -25 dB hitam hingga 0 dB putih)
    tile_sar_sebelum = safe_get_tile_url(
        s1FilteredSebelum.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    )
    tile_sar_sesudah = safe_get_tile_url(
        s1FilteredSesudah.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    )

    # Layer Flood Mask berwarna biru (#0066FF) hanya pada piksel bernilai banjir
    flood_vis = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(palette=['#0066FF'])
    tile_flood_mask = safe_get_tile_url(flood_vis)

    # Layer gabungan: SAR sesudah banjir ditumpuk dengan masker biru area genangan
    img_sar_sesudah_vis = s1FilteredSesudah.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    tile_sar_masked = safe_get_tile_url(img_sar_sesudah_vis.blend(flood_vis))

    # Layer Optik RGB Basemap Sentinel-2
    # [DINONAKTIFKAN] - Sentinel-2 RGB sementara tidak digunakan
    # tile_rgb_basemap = safe_get_tile_url(
    #     s2_basemap.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    # )

    # Layer RGB Masked (RGB + Overlay Genangan)
    # [DINONAKTIFKAN]
    # rgb_vis = s2_basemap.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    # tile_rgb_masked = safe_get_tile_url(rgb_vis.blend(flood_vis))

    # 12. Generate Thumbnail URLs (PNG statis 512 px) untuk kartu pratinjau visual dan fitur unduh gambar
    def safe_get_thumb_url(ee_image, region):
        """Mengambil URL gambar PNG statis berdimensi 512 px untuk pratinjau dan unduhan."""
        try:
            return ee_image.getThumbUrl({
                'region': region,
                'dimensions': 512,
                'format': 'png'
            })
        except Exception:
            return ""

    # Visualisasi untuk thumbnail
    sar_sebelum_vis = s1FilteredSebelum.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    sar_sesudah_vis = s1FilteredSesudah.select('VH').clip(roi).visualize(min=-25, max=0, palette=['black', 'white'])
    sar_masked_vis = img_sar_sesudah_vis.blend(flood_vis)
    # [DINONAKTIFKAN] - Sentinel-2 RGB sementara tidak digunakan
    # rgb_basemap_vis = s2_basemap.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)
    # rgb_masked_vis = rgb_vis.blend(flood_vis)
    flood_mask_vis = flood_vis

    thumb_sar_sebelum = safe_get_thumb_url(sar_sebelum_vis, roi)
    thumb_sar_sesudah = safe_get_thumb_url(sar_sesudah_vis, roi)
    thumb_sar_masked = safe_get_thumb_url(sar_masked_vis, roi)
    # [DINONAKTIFKAN] - Sentinel-2 RGB sementara tidak digunakan
    # thumb_rgb_basemap = safe_get_thumb_url(rgb_basemap_vis, roi)
    # thumb_rgb_masked = safe_get_thumb_url(rgb_masked_vis, roi)
    thumb_flood_mask = safe_get_thumb_url(flood_mask_vis, roi)

    # 13. Ekspor data vektor GeoJSON untuk batas wilayah (ROI) dan poligon genangan banjir
    roi_geojson = roi.getInfo()

    # Mengonversi raster floodMask menjadi poligon vektor pada skala 30 m agar ringan dirender browser
    flood_boundary_geojson = None
    try:
        flood_vectors = (floodMask.updateMask(floodMask)
                         .reduceToVectors(
                             geometry=roi,
                             scale=30,
                             maxPixels=1e8,
                             geometryType='polygon',
                             eightConnected=True,
                             labelProperty='flood'
                         ))
        flood_boundary_geojson = flood_vectors.getInfo()
    except Exception:
        flood_boundary_geojson = {"type": "FeatureCollection", "features": []}

    # Batas administrasi kecamatan Bandung Raya sebagai GeoJSON
    bandung_raya_geojson = wilayah_bandung_raya.getInfo()

    # 14. Titik tengah ROI (dibalik menjadi [lat, lon] mengikuti standar urutan koordinat Leaflet)
    roi_centroid = roi.centroid(1).coordinates().getInfo()
    map_center = [roi_centroid[1], roi_centroid[0]]

    # 15. Kotak pembatas (bounding box) ROI untuk fitur auto-fit layar peta di Leaflet
    roi_bounds_info = roi.bounds().getInfo()

    return {
        "luas_m2": luas_m2,
        "luas_ha": luas_m2 / 10000,
        "pre_period": f"{pre_start} s/d {pre_end}",
        "post_period": f"{flood_start} s/d {flood_end}",
        # GEE Tile URLs untuk Leaflet TileLayer (overlay interaktif)
        "tile_sar_sebelum": tile_sar_sebelum,
        "tile_sar_sesudah": tile_sar_sesudah,
        "tile_sar_masked": tile_sar_masked,
        "tile_flood_mask": tile_flood_mask,
        # [DINONAKTIFKAN] - Sentinel-2 RGB sementara tidak digunakan
        # "tile_rgb_basemap": tile_rgb_basemap,
        # "tile_rgb_masked": tile_rgb_masked,
        # Thumbnail URLs untuk perbandingan visual statis (download)
        "thumb_sar_sebelum": thumb_sar_sebelum,
        "thumb_sar_sesudah": thumb_sar_sesudah,
        "thumb_sar_masked": thumb_sar_masked,
        # [DINONAKTIFKAN] - Sentinel-2 RGB sementara tidak digunakan
        # "thumb_rgb_basemap": thumb_rgb_basemap,
        # "thumb_rgb_masked": thumb_rgb_masked,
        "thumb_flood_mask": thumb_flood_mask,
        # GeoJSON data untuk overlay
        "roi_geojson": roi_geojson,
        "flood_boundary_geojson": flood_boundary_geojson,
        "bandung_raya_geojson": bandung_raya_geojson,
        # Koordinat peta
        "map_center": map_center,
        "roi_bounds": roi_bounds_info,
    }
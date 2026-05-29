import ee
import datetime

def init_ee():
    """Inisialisasi koneksi ke Google Earth Engine."""
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')

def get_manual_geometry(kecamatan_name):
    """
    Menyimpan dan memanggil poligon buatan sendiri (Hanya area administrasi/ROI).
    """
    polygons = {
        "BandungRaya": ee.Geometry.Polygon([[[107.4877048585857, -6.801766035473307], [107.4877048585857, -7.08531528322579], [107.86810646991383, -7.08531528322579], [107.86810646991383, -6.801766035473307]]]),
        "Roi2": ee.Geometry.Polygon([[[107.51971278529716, -7.181286080060269], [107.51971278529716, -7.237486479851981], [107.58820567470146, -7.237486479851981], [107.58820567470146, -7.181286080060269]]]),
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
    # Mengembalikan poligon yang sesuai, jika tidak ada fallback ke Baleendah
    return polygons.get(kecamatan_name, polygons)


def get_flood_analysis(kecamatan_name, flood_start, flood_end):
    """
    Fungsi utama untuk mendeteksi banjir menggunakan Sentinel-1.
    """
    init_ee()

    # 1. Tentukan Region of Interest (ROI) menggunakan poligon manual
    roi = get_manual_geometry(kecamatan_name)
    
    # 2. Pengaturan Waktu Otomatis
    date_format = "%Y-%m-%d"
    f_start_date = datetime.datetime.strptime(flood_start, date_format)
    
    pre_start = (f_start_date - datetime.timedelta(days=30)).strftime(date_format)
    pre_end = (f_start_date - datetime.timedelta(days=1)).strftime(date_format)
    
    # 3. Panggil koleksi data Sentinel-1 GRD
    s1Collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(roi)
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))

    # 4. Ambil rata-rata (median)
    s1ImageSebelum = s1Collection.filterDate(pre_start, pre_end).median().clip(roi)
    s1ImageSesudah = s1Collection.filterDate(flood_start, flood_end).median().clip(roi)

    # 5. Speckle Filtering Sederhana (Focal Mean 30 meter)
    filterRadius = 30
    s1FilteredSebelum = s1ImageSebelum.focal_mean(filterRadius, 'circle', 'meters')
    s1FilteredSesudah = s1ImageSesudah.focal_mean(filterRadius, 'circle', 'meters')

    # 6. Perhitungan Backscatter Ratio (BR) pada band VH
    floodRatio = s1FilteredSesudah.select('VH').divide(s1FilteredSebelum.select('VH'))

    # 7. Penerapan Ambang Batas (Thresholding = 1.30)
    floodMask = floodRatio.gt(1.30)

    # 8. Pemurnian Hasil (Refinement)
    # a. Slope Masking (Kemiringan Lereng < 5%)
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))

    # b. Analisis Konektivitas (Menghilangkan noise terisolasi < 10 pixel)
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
    
    # Ekstrak nilai angka luas area
    luas_m2 = totalFloodArea.getInfo().get('VH', 0)
    if luas_m2 is None:
        luas_m2 = 0

    # 10. Visualisasi dan Ekspor ke Link Gambar (PNG)
    visVHSesudah = s1FilteredSesudah.select('VH').clip(roi).visualize(
        min=-25,
        max=0,
        palette=['black', 'white']
    )

    visFloodMask = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(
        palette=['blue']
    )

    # Menggabungkan gambar (Radar Grayscale + Masker Banjir Biru)
    overlayBanjir = visVHSesudah.blend(visFloodMask)

    paramsUnduh = {
        'dimensions': 1000,
        'region': roi,
        'format': 'png'
    }

    url_gambar = overlayBanjir.getThumbURL(paramsUnduh)

    return {
        "luas_m2": luas_m2,
        "luas_ha": luas_m2 / 10000,
        "url_gambar": url_gambar,
        "pre_period": f"{pre_start} s/d {pre_end}",
        "post_period": f"{flood_start} s/d {flood_end}"
    }
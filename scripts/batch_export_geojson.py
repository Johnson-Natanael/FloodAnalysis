"""
Batch Export GeoJSON Boundaries — Pre-generate flood boundary data untuk halaman Detail historis.

Script ini membaca data kejadian banjir dari CSV, menjalankan deteksi banjir GEE
untuk setiap kejadian, dan mengekspor boundary polygon genangan sebagai file GeoJSON.
"""

import ee
import json
import os
import time
import csv


def init_ee():
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='ee-johnsonnn')


def get_asset_geometry(nama_area):
    # Ambil geometri ROI dari Asset GEE.
    batas_wilayah = ee.FeatureCollection('projects/ee-johnsonnn/assets/Bandung_Raya')
    daftar_kabkota = ['Bandung', 'Bandung Barat', 'Kota Bandung', 'Kota Cimahi', 'Cimahi']
    wilayah_utama = batas_wilayah.filter(ee.Filter.inList('WADMKK', daftar_kabkota))

    daftar_sumedang_kc = ['Cimanggung', 'Tanjungsari', 'Sukasari', 'Jatinangor', 'Pamulihan']
    sumedang_barat = (batas_wilayah
                      .filter(ee.Filter.inList('WADMKK', ['Sumedang', 'SUMEDANG', 'Kabupaten Sumedang']))
                      .filter(ee.Filter.inList('WADMKC', daftar_sumedang_kc)))
    
    wilayah_bandung_raya = wilayah_utama.merge(sumedang_barat)

    if nama_area.startswith("BandungRaya"):
        return wilayah_bandung_raya.geometry().dissolve(), wilayah_bandung_raya

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

    return roi_feature.geometry(), wilayah_bandung_raya


def detect_flood_and_export(roi, pre_start, pre_end, post_start, post_end, scale_vec=30):
    """
    Jalankan deteksi banjir dan kembalikan:
    - flood_geojson: FeatureCollection GeoJSON dari area genangan
    - roi_geojson: Geometry GeoJSON dari ROI
    - luas_m2: Luas area terdampak dalam m²
    """
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

    # Pemurnian: Slope masking + Connected pixels
    dem = ee.Image('USGS/SRTMGL1_003')
    slope = ee.Terrain.slope(dem)
    floodMask = floodMask.updateMask(slope.lt(5))
    connections = floodMask.connectedPixelCount(100)
    floodMask = floodMask.updateMask(connections.gte(10))

    # Hitung luas
    floodArea = floodMask.multiply(ee.Image.pixelArea())
    totalFloodArea = floodArea.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e9
    )
    luas_m2 = totalFloodArea.getInfo().get('VH', 0) or 0

    # Ekspor sebagai vektor GeoJSON
    flood_geojson = {"type": "FeatureCollection", "features": []}
    try:
        flood_vectors = (floodMask.updateMask(floodMask)
                         .reduceToVectors(
                             geometry=roi,
                             scale=scale_vec,
                             maxPixels=1e8,
                             geometryType='polygon',
                             eightConnected=True,
                             labelProperty='flood'
                         ))
        flood_geojson = flood_vectors.getInfo()
    except Exception as e:
        print(f"      [WARN] Gagal reduceToVectors: {e}")

    roi_geojson = roi.getInfo()

    return flood_geojson, roi_geojson, luas_m2


def save_geojson(data, filepath):
    """Simpan dict sebagai file GeoJSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"      -> Tersimpan: {filepath} ({size_kb:.1f} KB)")


def main():
    init_ee()

    # Baca CSV kejadian historis
    csv_path = os.path.join('web_app', 'data_banjir_historis.csv')
    output_dir = os.path.join('web_app', 'static', 'assets', 'geojson')
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        events = list(reader)

    print(f"Memulai batch export GeoJSON untuk {len(events)} kejadian...\n")

    for i, event in enumerate(events, 1):
        ev_id = event['id']
        daerah = event['daerah']
        kode_raya = event['kode_raya']
        tahun = event['tahun']
        pre_start = event['pre_start']
        pre_end = event['pre_end']
        post_start = event['post_start']
        post_end = event['post_end']

        print(f"[{i}/{len(events)}] ID={ev_id} | {daerah} (Tahun {tahun})")
        print(f"    Pre: {pre_start} s/d {pre_end}")
        print(f"    Post: {post_start} s/d {post_end}")

        # Fokus (per kecamatan)
        print(f"    [A] Fokus: {daerah}")
        try:
            roi_fokus, _ = get_asset_geometry(daerah)
            flood_fokus, roi_fokus_geo, luas_fokus = detect_flood_and_export(
                roi_fokus, pre_start, pre_end, post_start, post_end, scale_vec=10
            )
            save_geojson(flood_fokus, os.path.join(output_dir, f"{ev_id}_{daerah}_flood.geojson"))
            save_geojson(roi_fokus_geo, os.path.join(output_dir, f"{ev_id}_{daerah}_roi.geojson"))
            print(f"      Luas fokus: {luas_fokus:.2f} m²")
        except Exception as e:
            print(f"      [ERROR] Fokus {daerah}: {e}")

        # Regional (BandungRaya)
        print(f"    [B] Regional: {kode_raya}")
        try:
            roi_raya, _ = get_asset_geometry(kode_raya)
            flood_raya, roi_raya_geo, luas_raya = detect_flood_and_export(
                roi_raya, pre_start, pre_end, post_start, post_end
            )
            save_geojson(flood_raya, os.path.join(output_dir, f"{ev_id}_{kode_raya}_flood.geojson"))
            save_geojson(roi_raya_geo, os.path.join(output_dir, f"{ev_id}_{kode_raya}_roi.geojson"))
            print(f"      Luas regional: {luas_raya:.2f} m²")
        except Exception as e:
            print(f"      [ERROR] Regional {kode_raya}: {e}")

        print()
        # Jeda 2 detik agar tidak kena limit API GEE
        time.sleep(2)

    print("=" * 60)
    print("Batch export selesai")
    print(f"File GeoJSON tersimpan di: {output_dir}")


if __name__ == "__main__":
    main()

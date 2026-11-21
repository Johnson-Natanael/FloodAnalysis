"""
Export Raster Flood Mask PNG Menghasilkan file PNG transparan untuk genangan banjir historis.
"""

import os
import csv
import time
import requests
import ee


def init_ee():
    try:
        ee.Initialize(project='ee-johnsonnn')
    except Exception:
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


def generate_and_download_floodmask(roi, pre_start, pre_end, post_start, post_end, filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 500:
        print(f"    -> Sudah ada (lewati): {os.path.basename(filepath)}")
        return True

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

    # Visualisasi floodmask: hanya pixel bernilai 1 yang berwarna biru #0066FF
    visFloodMask = floodMask.toInt().clip(roi).updateMask(floodMask).visualize(palette=['#0066FF'])

    paramsUnduh = {
        'dimensions': 1000,
        'region': roi.bounds(),
        'format': 'png'
    }

    try:
        url = visFloodMask.getThumbURL(paramsUnduh)
    except Exception as e:
        print(f"    -> Gagal membuat ThumbURL: {e}")
        return False

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(resp.content)
                size_kb = os.path.getsize(filepath) / 1024
                print(f"    -> Berhasil disimpan: {os.path.basename(filepath)} ({size_kb:.1f} KB)")
                return True
            else:
                print(f"    -> HTTP {resp.status_code} pada percobaan {attempt}...")
        except Exception as e:
            print(f"    -> Error percobaan {attempt}: {e}")
        time.sleep(3)

    return False


def main():
    init_ee()

    # Cari file CSV data historis
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'web_app', 'data_banjir_historis.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, 'data_banjir_historis.csv')

    output_dir = os.path.join(base_dir, 'web_app', 'static', 'assets', 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        events = list(reader)

    print(f"Memulai ekspor raster PNG flood mask untuk {len(events)} kejadian banjir...\n")

    for i, ev in enumerate(events, 1):
        ev_id = ev['id']
        tahun = ev['tahun']
        daerah = ev['daerah']
        kode_raya = ev['kode_raya']
        pre_start = ev['pre_start']
        pre_end = ev['pre_end']
        post_start = ev['post_start']
        post_end = ev['post_end']

        daerah_safe = daerah.replace(" ", "")
        kode_raya_safe = kode_raya.replace(" ", "")

        print(f"[{i}/{len(events)}] ID {ev_id}: {daerah} & {kode_raya} ({tahun})")

        # 1. Fokus
        file_fokus = os.path.join(output_dir, f"{tahun}_flood_mask_{daerah_safe}.png")
        try:
            roi_fokus = get_asset_geometry(daerah)
            generate_and_download_floodmask(roi_fokus, pre_start, pre_end, post_start, post_end, file_fokus)
        except Exception as e:
            print(f"    -> [ERROR] Fokus {daerah}: {e}")

        # 2. Regional Bandung Raya
        file_raya = os.path.join(output_dir, f"{tahun}_flood_mask_{kode_raya_safe}.png")
        try:
            roi_raya = get_asset_geometry(kode_raya)
            generate_and_download_floodmask(roi_raya, pre_start, pre_end, post_start, post_end, file_raya)
        except Exception as e:
            print(f"    -> [ERROR] Regional {kode_raya}: {e}")

        time.sleep(1)

    print("\n" + "=" * 60)
    print("Selesai! Seluruh raster PNG flood mask telah tersimpan di:")
    print(output_dir)


if __name__ == "__main__":
    main()

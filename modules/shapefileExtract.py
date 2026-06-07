import geopandas as gpd

# Arahkan path ke lokasi file .shp Anda di komputer
file_path = r"C:\Users\johns\Documents\tugasKuliah\TA\BatasWilayahKecamatan\LapakGIS_Batas_Kecamatan_2024.shp"

# Membaca shapefile menjadi GeoDataFrame
gdf = gpd.read_file(file_path)

# Menampilkan informasi ringkas struktur kolom tabel atribut
print(gdf.info())

# Menghitung koordinat kotak pembatas spasial untuk objek baris pertama
polygon_pertama = gdf['geometry'].iloc[0]
bbox_coords = polygon_pertama.bounds
print("Koordinat Kotak Pembatas Objek (BBOX):", bbox_coords)

df_bandung = gdf[gdf['WADMKK'].str.contains('BANDUNG', case=False, na=False)]

print("=== KABUPATEN/KOTA YANG TERSARING ===")
daftar_wadmkk = df_bandung['WADMKK'].unique()
print(daftar_wadmkk)

# Agregasi: Menghitung jumlah kecamatan per Kabupaten/Kota
print("\n=== JUMLAH KECAMATAN PER WILAYAH ===")
agregat_jumlah = df_bandung.groupby('WADMKK')['WADMKC'].count()
print(agregat_jumlah)

# Menampilkan daftar lengkap Kecamatan (WADMKC) untuk setiap wilayah
print("\n=== DAFTAR KECAMATAN ===")
for wilayah in daftar_wadmkk:
    kecamatan_list = df_bandung[df_bandung['WADMKK'] == wilayah]['WADMKC'].tolist()
    print(f"\nWilayah: {wilayah}")
    print(kecamatan_list)


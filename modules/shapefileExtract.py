import geopandas as gpd

# 1. Arahkan path ke lokasi file .shp Anda di komputer
file_path = r"C:\Users\johns\Documents\tugasKuliah\TA\BatasWilayahKecamatan\LapakGIS_Batas_Kecamatan_2024.shp"

# Membaca shapefile menjadi GeoDataFrame
gdf = gpd.read_file(file_path)

df_bandung = gdf[gdf['WADMKK'].str.contains('BANDUNG', case=False, na=False)]

print("=== KABUPATEN/KOTA YANG TERSARING ===")
daftar_wadmkk = df_bandung['WADMKK'].unique()
print(daftar_wadmkk)

# 4. Agregasi: Menghitung jumlah kecamatan per Kabupaten/Kota
print("\n=== JUMLAH KECAMATAN PER WILAYAH ===")
agregat_jumlah = df_bandung.groupby('WADMKK')['WADMKC'].count()
print(agregat_jumlah)

# 5. Menampilkan daftar lengkap Kecamatan (WADMKC) untuk setiap wilayah
print("\n=== DAFTAR KECAMATAN ===")
for wilayah in daftar_wadmkk:
    kecamatan_list = df_bandung[df_bandung['WADMKK'] == wilayah]['WADMKC'].tolist()
    print(f"\nWilayah: {wilayah}")
    print(kecamatan_list)
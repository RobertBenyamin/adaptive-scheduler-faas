import pandas as pd
import os

def parse_pivot_sheet(file_path):
    """
    Fungsi ini membaca dan mem-parsing sheet 'Pivot_Tables' 
    dari satu file Excel.
    (Fungsi ini SAMA seperti sebelumnya, tidak ada perubahan)
    """
    try:
        df_raw = pd.read_excel(file_path, sheet_name='Pivot_Tables', header=None)
    except FileNotFoundError:
        print(f"PERINGATAN: File {file_path} tidak ditemukan. File ini akan dilewati.")
        return None
    except Exception as e:
        print(f"ERROR: Gagal membaca {file_path}: {e}. File ini akan dilewati.")
        return None

    all_data = []
    current_service = None
    header = None
    service_data = []

    for index, row in df_raw.iterrows():
        cols = [str(item) if not pd.isna(item) else "" for item in row]
        
        if all(c == "" for c in cols):
            continue
        
        if "SERVICE" in cols[0] and all(c == "" for c in cols[1:]):
            if current_service and header and service_data:
                df_service = pd.DataFrame(service_data, columns=header)
                df_service['Service'] = current_service
                all_data.append(df_service)
            
            current_service = cols[0].replace(" SERVICE", "")
            service_data = []
            header = None
            continue
        
        if "Statistic" in cols[0]:
            header = cols
            continue
            
        if current_service and header:
            if len(cols) == len(header):
                service_data.append(cols)

    if current_service and header and service_data:
        df_service = pd.DataFrame(service_data, columns=header)
        df_service['Service'] = current_service
        all_data.append(df_service)

    if not all_data:
        print(f"PERINGATAN: Tidak ada data pivot yang dapat diparsing dari {file_path}.")
        return None

    df_combined = pd.concat(all_data, ignore_index=True)
    df_cleaned = df_combined[df_combined['Statistic'].str.strip() != ""].copy()

    numeric_cols = ['LOW_LOAD', 'MED_LOAD', 'HIGH_LOAD']
    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
    return df_cleaned

def main():
    """
    Fungsi utama untuk me-loop 10 file, memprosesnya, 
    dan menyimpan hasilnya.
    (Fungsi ini telah DIMODIFIKASI)
    """
    all_trials_data = []
    
    print("Mulai memproses file...")

    file_title = "runners5"
    trial_title = "AVG"
    file_type = "5s051020"
    
    for i in range(1, 11):
        file_name = f"{file_title}-{file_type}-{i}.xlsx"
        
        # --- PERUBAHAN 1 ---
        # Nama 'Trial' sekarang adalah nama file (sesuai permintaan Anda)
        trial_name = f"{file_title}-{file_type}-{i}" 
        
        print(f"Memproses {file_name}...")
        
        df_trial = parse_pivot_sheet(file_name)
        
        if df_trial is not None:
            df_trial['Trial'] = trial_name # Tambahkan kolom 'Trial'
            all_trials_data.append(df_trial)
    
    if not all_trials_data:
        print("ERROR: Tidak ada data yang berhasil diproses. Program berhenti.")
        return

    # --- 1. Membuat File Gabungan (sesuai contoh Anda) ---
    print("\nMembuat file gabungan 'combined_all_trials.csv'...")
    
    df_all_data = pd.concat(all_trials_data, ignore_index=True)
    
    try:
        # Urutan kolom untuk file gabungan
        cols_order_combined = ['Service', 'Trial', 'Statistic', 'LOW_LOAD', 'MED_LOAD', 'HIGH_LOAD']
        df_all_data_ordered = df_all_data[cols_order_combined]
    except Exception as e:
        print(f"Peringatan (Gabungan): Gagal menyusun ulang kolom. Menyimpan dengan urutan default. Error: {e}")
        df_all_data_ordered = df_all_data

    output_file_combined = "combined_all_trials.csv"
    df_all_data_ordered.to_csv(output_file_combined, index=False)
    print(f"SUKSES! Data gabungan disimpan ke: {output_file_combined}")

    # --- 2. Membuat File Rata-rata (sesuai teks permintaan Anda) ---
    print("\nMenghitung rata-rata dan membuat file 'average_all_trials.csv'...")

    # Tentukan urutan kustom untuk 'Statistic'
    # Ini adalah urutan yang Anda inginkan (dari file combined)
    stat_order = ['Min', 'Max', 'Mean', 'Median', 'P90', 'P95', 'P99', 'Std Dev']
    
    # Ubah kolom 'Statistic' di DataFrame utama menjadi tipe Kategori
    # Ini memberitahu pandas untuk *selalu* menggunakan urutan ini, bukan alfabet
    try:
        df_all_data['Statistic'] = pd.Categorical(
            df_all_data['Statistic'], 
            categories=stat_order, 
            ordered=True
        )
    except Exception as e:
        print(f"Peringatan: Gagal menerapkan urutan kustom pada 'Statistic'. Error: {e}")
        # Jika ada statistik yang tidak dikenal, program tetap lanjut
        pass
    
    # Hitung rata-rata (dikelompokkan berdasarkan Service dan Statistic)
    df_average = df_all_data.groupby(['Service', 'Statistic'], observed=False)[['LOW_LOAD', 'MED_LOAD', 'HIGH_LOAD']].mean().reset_index()
    
    # --- PERUBAHAN 2 ---
    df_average['Trial'] = trial_title
    
    # --- PERUBAHAN 3 ---
    # Susun ulang kolom untuk file rata-rata
    try:
        cols_order_average = ['Service', 'Trial', 'Statistic', 'LOW_LOAD', 'MED_LOAD', 'HIGH_LOAD']
        df_average_ordered = df_average[cols_order_average]
    except Exception as e:
        print(f"Peringatan (Rata-rata): Gagal menyusun ulang kolom. Menyimpan dengan urutan default. Error: {e}")
        df_average_ordered = df_average

    # Simpan file rata-rata ke CSV
    output_file_average = "average_all_trials.csv"
    df_average_ordered.to_csv(output_file_average, index=False)
    print(f"SUKSES! Data rata-rata disimpan ke: {output_file_average}")
    print("\nSemua proses selesai.")

# Menjalankan fungsi utama
if __name__ == "__main__":
    main()
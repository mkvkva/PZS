import pandas as pd
import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt

sampfrom = 0  # Počáteční index vzorku
nsamp = 10  # Počet vzorků
lowcut = 0.5  # Dolní hranice pásmové filtrace (Hz)
highcut = 2.211  # Horní hranice pásmové filtrace (Hz)
filter_order = 4  # Pořadí filtru
threshold = 0.0001  # Práh pro detekci R vrcholů

# Automatické zavedení a vytvoření složky pro data
data_dir = "../data/real_ekg"
os.makedirs(data_dir, exist_ok=True)

# Pokud složka je prázdná, stáhneme data automaticky
if not os.listdir(data_dir):
    print("Záznamy nebyly nalezeny. Stahování z PhysioNet...")
    wfdb.dl_database("butqdb", dl_dir=data_dir)
    print("✅ Stahování dokončeno!")

def bandpass_filter(signal, lowcut, highcut, fs, order):
    """Funkce pro pásmovou filtrační metodu
    Args:
        signal: vstupní signál, který má být filtrován.
        lowcut: dolní mez frekvenčního pásma
        highcut: horní mez frekvenčního pásma.
        fs: vzorkovací frekvence signálu.
        order: řád filtru, který určuje strmost filtru.
    Returns:
        Filtrovaný signál.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def calculate_heart_rate(peaks, sampling_frequency):
    """Funkce pro výpočet tepové frekvence (BPM) z detekovaných R vrcholů.
    Args:
        peaks: indexy detekovaných R vrcholů.
    Returns:
        Tepová frekvence v BPM.
    """
    if peaks.size > 1:
        r_peak_intervals = np.diff(peaks) / sampling_frequency
        return 60 / np.mean(r_peak_intervals) if r_peak_intervals.size > 0 else 0
    else:
        return 0

def get_data():
    """Funkce pro načtení dat a zpracování EKG signálů."""
    data = []
    
    # Načtení názvů souborů ve složce real_ekg
    filenames = {file.split(".")[0] for file in os.listdir(data_dir) if file.endswith(".dat")}
    print(f"Počet souborů: {len(filenames)}")
    
    for name in filenames:
        # Sestavení cesty k souboru
        record_path = os.path.join(data_dir, name)
        
        # Načtení signálu a informací o signálu
        record = wfdb.rdsamp(record_path, sampfrom=sampfrom, sampto=sampfrom + nsamp)
        
        # Načtení referenčních R vrcholů
        true_peaks = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=sampfrom + nsamp).sample
        
        # Signál
        signal = record[0][:, 0]
        fields = record[1]
        
        # Vzorkovací frekvence (Hz)
        sampling_frequency = fields["fs"]
        
        # Normalizace signálu
        signal_normalized = (signal - np.mean(signal)) / np.std(signal)
        
        # Aplikace filtrace
        try:
            signal_processed = bandpass_filter(signal_normalized, lowcut, highcut, sampling_frequency, filter_order)
        except ValueError as e:
            print(f"Chyba filtrace: {e}")
            signal_processed = signal_normalized
        
        # Derivace signálu
        derivative = np.gradient(signal_processed)
        
        # Hledání nulových průchodů podle derivace
        zero_crossings = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0]
        
        # Filtrace nulových průchodů podle prahu
        peaks = zero_crossings[(signal_processed[zero_crossings] > threshold) & (zero_crossings < len(signal_processed))]
        
        # Přidání výsledků do seznamu
        data.append([
            name,
            calculate_heart_rate(peaks, sampling_frequency),
            calculate_heart_rate(true_peaks, sampling_frequency),
        ])
    
    # Vytvoření DataFrame
    df = pd.DataFrame(data, columns=["name", "heart_rate", "true_heart_rate"])
    # Výpočet přesnosti
    df["accuracy"] = (1 - abs(df["heart_rate"] - df["true_heart_rate"]) / df["true_heart_rate"]) * 100
    return df

if __name__ == "__main__":
    df = get_data()
    print(df)
    print(f"Průměrná přesnost: {df['accuracy'].mean()} %")

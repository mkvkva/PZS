import pandas as pd
import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt

sampfrom = 0  # Počáteční index vzorku
nsamp = 100000  # Počet vzorků
lowcut = 0.5  # Dolní hranice pásmové filtrace (Hz)
highcut = 2.211  # Horní hranice pásmové filtrace (Hz)
filter_order = 4  # Pořadí filtru
threshold = 0.0001  # Práh pro detekci R vrcholů


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
        # Výpočet intervalů mezi R vrcholy
        # np.diff(peaks) vypočítá rozdíly mezi po sobě jdoucími indexy vrcholů, což představuje počet vzorků mezi vrcholy.
        # Dělením vzorkovací frekvencí (sampling_frequency) se tyto rozdíly převedou na časové intervaly v sekundách.
        r_peak_intervals = np.diff(peaks) / sampling_frequency
        # Výpočet průměrné tepové frekvence:
        # np.mean(r_peak_intervals) vypočítá průměrný interval mezi R vrcholy.
        # Tepová frekvence (BPM) je rovna 60 děleno průměrným intervalem mezi R vrcholy.
        # Pokud nebyly detekovány žádné R vrcholy, je tepová frekvence nastavena na 0.
        return 60 / np.mean(r_peak_intervals) if r_peak_intervals.size > 0 else 0
    else:
        return 0


def get_data():
    # Načtení dat
    data: list = []

    # Načtení názvů souborů
    filenames = set()
    main_directory = "../data/real_ekg"
    for idx, file in enumerate(os.listdir(main_directory)):
        # if idx == 20:
        #     break
        if file.endswith(".dat"):
            filenames.add(file.split(".")[0])

    print(f"Počet souborů: {len(filenames)}")
    # Procházení souborů
    for name in filenames:
        # Načtení signálu a informací o signálu
        record_path = os.path.join(main_directory, name)

        # Načtení signálu a informací o signálu
        record = wfdb.rdsamp(record_path, sampfrom=sampfrom, sampto=sampfrom + nsamp)

        # Načtení referenčních R vrcholů
        true_peaks = wfdb.rdann(
            record_path, extension="atr", sampfrom=sampfrom, sampto=sampfrom + nsamp
        ).sample

        # Signál
        signal = record[0][:, 0]
        fields = record[1]

        # Vzorkovací frekvence (Hz)
        sampling_frequency = fields["fs"]

        # Normalizace signálu
        # Tak, aby měl nulový průměr a jednotkovou směrodatnou odchylku
        signal_normalized = (signal - np.mean(signal)) / np.std(signal)

        # Aplikace filtrace
        try:
            signal_processed = bandpass_filter(
                signal_normalized, lowcut, highcut, sampling_frequency, filter_order
            )
        except ValueError as e:
            print(f"Chyba filtrace: {e}")
            signal_processed = signal_normalized

        # Derivace signálu je užitečná pro detekci změn v signálu, jako jsou vrcholy a průchody nulou.
        # V kontextu EKG signálu se derivace používá k detekci R vrcholů, které odpovídají srdečním úderům.
        derivative = np.gradient(signal_processed)

        # Hledání nulových průchodů podle derivace
        zero_crossings = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0]

        # Filtrace nulových průchodů podle prahu
        peaks = zero_crossings[
            (signal_processed[zero_crossings] > threshold)
            & (zero_crossings < len(signal_processed))
        ]
        # Přidání výsledků do seznamu
        # Název souboru, vypočítaná tepová frekvence, referenční tepová frekvence
        data.append(
            [
                name,
                calculate_heart_rate(peaks, sampling_frequency),
                calculate_heart_rate(true_peaks, sampling_frequency),
            ]
        )
    # Vytvoření DataFrame
    df = pd.DataFrame(data, columns=["name", "heart_rate", "true_heart_rate"])
    # Výpočet přesnosti
    df["accuracy"] = (
        1 - abs(df["heart_rate"] - df["true_heart_rate"]) / df["true_heart_rate"]
    ) * 100
    return df


if __name__ == "__main__":
    df: pd.DataFrame = get_data()
    print(df)
    print(f"Průměrná přesnost: {df['accuracy'].mean()} %")

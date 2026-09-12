"""Generate file audio (MP3) utk tiap pesan peringatan kesalahan postur --
bab 8.3.3 proposal ("teks dan suara peringatan"), gaya sama seperti Ko et al.
(audio PRE-RECORDED, bukan TTS live saat runtime -- lihat Streamlit.py mereka
yg pakai file .mp3 siap pakai + pygame.mixer, mis. "excessive_arch_1.mp3").

BEDA dgn Ko et al.: mereka merekam/menyiapkan file audio-nya sendiri (metode
pastinya tidak diketahui -- mungkin rekaman suara asli). Kita generate SEKALI
via gTTS (Google Text-to-Speech, https://pypi.org/project/gTTS/) lalu
menyimpan hasilnya sbg file .mp3 di assets/audio/ -- setelah itu app.py HANYA
memutar file yg sudah jadi (st.audio(..., autoplay=True)), TIDAK pernah
memanggil gTTS saat aplikasi jalan (proposal + diskusi proyek: generate
live/TTS runtime "lebih berat", jadi harus pre-recorded).

Jalankan ulang skrip ini HANYA kalau teks pesan di warnings_content.py
berubah. Perlu koneksi internet (gTTS memanggil Google Translate TTS API).

Jalankan:
    python scripts/generate_warning_audio.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gtts import gTTS

from src.app.warnings_content import WARNING_MESSAGES

AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for key, message in WARNING_MESSAGES.items():
        out_path = AUDIO_DIR / f"{key}.mp3"
        print(f"Generating {out_path.name} <- \"{message}\"")
        tts = gTTS(text=message, lang="id")
        tts.save(str(out_path))
    print(f"\nSelesai: {len(WARNING_MESSAGES)} file audio tersimpan di {AUDIO_DIR}")


if __name__ == "__main__":
    main()

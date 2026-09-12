"""Teks + audio peringatan kesalahan postur -- bab 8.3.3 proposal, gaya sama
seperti Ko et al. (st.error() + audio pre-recorded, bukan TTS live). Audio
digenerate sekali (scripts/generate_warning_audio.py, gTTS), disimpan di
assets/audio/, diputar ulang oleh app.py.

Kelas "correct" tidak punya pesan/audio. 9 kombinasi (3 exercise x 3 postur)
-- fase concentric/eccentric tidak dibedakan pesannya. Nama kelas postur
persis sama dengan filename_parser.py."""
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "audio"

WARNING_MESSAGES = {
    "squat_kneeinward": "Jaga lutut tetap sejajar ujung kaki, jangan sampai masuk ke dalam",
    "squat_backbend": "Jaga punggung tetap tegak, jangan membungkuk ke depan",
    "deadlift_backround": "Pertahankan punggung lurus/netral, jangan membulat",
    "deadlift_armsspread": "Posisikan tangan selebar bahu, jangan terlalu lebar",
    "benchpress_flatback": "Pertahankan sedikit lengkungan alami di punggung bawah, jangan menempel rata ke bench",
    "benchpress_armsspread": "Jaga siku tidak terlalu melebar ke samping saat menurunkan beban",
}


def get_warning_key(predicted_class):
    """predicted_class: string lengkap dari model, mis.
    "benchpress_armsspread_concentric" -- potong suffix fase
    ("_concentric"/"_eccentric") jadi key spt "benchpress_armsspread".
    Return None kalau kelasnya "correct" (tidak perlu peringatan) atau
    predicted_class None (belum ada window penuh)."""
    if predicted_class is None:
        return None
    key = predicted_class.rsplit("_", 1)[0]
    if "_correct" in key:
        return None
    return key


def get_warning(predicted_class):
    """Teks peringatan (Indonesia) utk ditampilkan (st.error/overlay teks di
    video). None kalau tidak ada peringatan (lihat get_warning_key)."""
    key = get_warning_key(predicted_class)
    if key is None:
        return None
    return WARNING_MESSAGES.get(key)


def get_warning_audio_path(predicted_class):
    """Path file .mp3 pre-recorded (assets/audio/<key>.mp3, lihat
    scripts/generate_warning_audio.py) utk predicted_class ybs, atau None
    kalau tidak ada peringatan / filenya belum digenerate."""
    key = get_warning_key(predicted_class)
    if key is None:
        return None
    path = AUDIO_DIR / f"{key}.mp3"
    return path if path.exists() else None

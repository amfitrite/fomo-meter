# Arsitektur FOMO Meter

Dokumen ini menjelaskan alur teknis aplikasi desktop **FOMO Meter**. Fokusnya
adalah pemisahan data, inferensi, verifikasi, dan presentasi agar status sumber
selalu jujur.

## 1. Lapisan sistem

```text
Sumber publik
  StockTwits / Google News / Yahoo RSS / Reddit / X (token)
        │
        ▼
Adapter + cache + timeout + status per sumber
        │
        ▼
Relevance gate ──> lexicon deterministik ──> SLM-1 (jika yakin)
        │
        ▼
Consensus + conflict handling + 9 integrity gates
        │
        ├──> engine.py (scan, brief, history, market context)
        │
        ▼
HTTP server lokal 127.0.0.1:<ephemeral>
        │
        ├──> HTML/CSS/JS dashboard
        └──> PySide6 + QtWebEngine desktop shell
```

- **Adapter sumber** hanya mengembalikan payload nyata atau status gagal. Tidak
  ada fallback sintetis.
- **Relevance gate** menolak spam, teks non-finansial, dan instruksi injeksi.
- **Klasifikasi** memakai lexicon sebagai baseline/fallback. SLM-1 dipakai bila
  tersedia dan keyakinannya memadai.
- **Konsensus** tidak merata-ratakan BUY dan SELL menjadi satu angka. Konflik
  ditampilkan sebagai `CONFLICT`.
- **Konteks pasar** (harga, Fear & Greed) hanya untuk konteks; tidak boleh
  mengubah arah sentimen.

## 2. SLM-1

- Fitur: hashed word/char n-gram, 2.048 dimensi.
- Jaringan: 2.048 → 128 ReLU → 64 ReLU → softmax 3 kelas (`BUY`, `SELL`, `WAIT`).
- Pelatihan: numpy murni, Adam, dropout, class weighting, early stopping.
- Dataset: 1.042 contoh kurasi; 885 train / 157 validation.
- Validasi: akurasi 94,27%; macro F1 0,944.
- Bobot lokal: `models/slm1.npz`; metadata: `models/slm1_meta.json`.

SLM-1 adalah classifier saraf, bukan model generatif. `brief` adalah ringkasan
deterministik dari output model dan bukti post yang lolos gate, bukan teks bebas.

## 3. Anti-halusinasi

Sembilan gate di `anti_halu_gates.py` memeriksa:

1. sumber live tersedia;
2. simbol sesuai ticker;
3. relevansi finansial;
4. kecukupan sampel;
5. kesegaran data;
6. bukti verbatim;
7. konflik tidak dirata-ratakan;
8. provenance URL;
9. tidak ada data sintetis.

Jika gate gagal, UI menampilkan status dan alasan. Aplikasi tidak mengisi celah
dengan data karangan.

## 4. Desktop shell frameless

`desktop.py` menjalankan server lokal lalu membuka
`http://127.0.0.1:<port>/#desktop` di `QWebEngineView`.

- `QMainWindow` memakai `Qt.FramelessWindowHint`; title bar native dihilangkan.
- Top bar web berfungsi sebagai custom chrome: minimize, maximize/restore,
  close, dan area drag.
- Halaman hanya menampilkan kontrol jendela jika hash `#desktop` ada. Dashboard
  browser biasa tidak menampilkan kontrol palsu.
- Perintah jendela dikirim lewat bridge console yang sempit:
  `FOMO_WINDOW|minimize`, `toggle_max`, `close`, `drag_start`, `drag_move`,
  `drag_end`. `AppWebPage` hanya menerima prefix ini dari origin lokal.
- Drag memakai `QWindow.startSystemMove()` agar snapping dan animasi Windows
  tetap native; tersedia fallback manual berbasis `QCursor`.
- Resize dari sudut kanan bawah memakai `QSizeGrip`, karena frameless window
  kehilangan border resize native.
- Di Windows 11, `DwmSetWindowAttribute` memberi sudut membulat. Kegagalan DWM
  diabaikan dengan aman.
- Tautan bukti eksternal dibuka di browser default; jendela aplikasi tidak
  pernah berpindah dari dashboard lokal.

## 5. FPS meter

FPS meter di kanan atas dihitung dari `requestAnimationFrame`:

- frame dihitung selama jendela 500 ms;
- hasil dibulatkan dan ditampilkan sebagai `NN FPS`;
- warna berubah menjadi amber di bawah 50 FPS dan merah di bawah 30 FPS;
- angka tidak pernah di-hardcode.

Karena memakai vsync compositor, FPS mengikuti refresh rate layar. Pada layar
60 Hz, nilai stabil adalah 60 FPS; pada layar 120 Hz, nilai dapat mencapai 120
FPS. Jika hardware acceleration dimatikan, Chromium dapat membatasi animasi dan
meter akan menampilkan angka rendah secara jujur.

## 6. Batas keamanan dan data

- Server hanya bind ke `127.0.0.1` dan memakai port ephemeral.
- Dashboard tidak memakai CDN, tracker, atau aset pihak ketiga.
- Semua model dan sprite berada di folder proyek.
- Tidak ada Ollama, OpenAI, HuggingFace, atau API model hosted.
- Sumber yang gagal tetap terlihat sebagai `ERROR`, `NO_DATA`, atau
  `NEEDS_AUTH`.

## 7. Verifikasi

```powershell
python -m unittest discover -s tests -v
python tools/screenshot.py
python tools/install_shortcut.py
```

Shortcut target: `pythonw.exe "D:\Sentiment LLM\run_gui.pyw"` dengan working
directory `D:\Sentiment LLM`.

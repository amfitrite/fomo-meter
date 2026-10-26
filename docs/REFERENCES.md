# Referensi GitHub

Riset referensi: **26 Oktober 2026**. Jumlah bintang bersifat dinamis dan hanya
menunjukkan popularitas, bukan kualitas. Proyek ini memakai referensi untuk
**ide arsitektur, pola integrasi, dan pelajaran desain**, bukan menyalin kode.
Periksa lisensi masing-masing repo sebelum memakai kode secara langsung.

## Ringkasan: yang paling relevan

| Repo | Fokus | Bintang | Lisensi | Yang relevan untuk proyek ini |
|---|---|---:|---|---|
| [breathworklabs/sentiment-scanner](https://github.com/breathworklabs/sentiment-scanner) | StockTwits + Reddit + VADER/FinBERT + dashboard | 1 | MIT | Arsitektur paling dekat: multi-sumber, status per sumber, skor sentimen, dashboard. Menjadi acuan untuk kartu sumber dan penanganan data kosong. |
| [AI4Finance-Foundation/FinNLP](https://github.com/AI4Finance-Foundation/FinNLP) | Pipeline data finansial | 1.496 | MIT | Pemisahan lapisan ingestion, normalisasi, dan analisis. Acuan struktur adapter sumber. |
| [ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) | Klasifikasi sentimen finansial berbasis BERT | 2.264 | Apache-2.0 | Taksonomi sentimen finansial dan pentingnya domain-specific language. Bobotnya tidak dipakai karena model harus buatan sendiri. |
| [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Financial LLM dan data pipeline | 21.470 | MIT | Gagasan memisahkan model, data, dan evaluasi; penekanan pada data provenance. Tidak memakai model atau API eksternal. |
| [fa33az/equitymood](https://github.com/fa33az/equitymood) | Dashboard sentimen finansial Indonesia | 1 | Tidak ada | Referensi terdekat untuk bilingual Indonesia/Inggris dan konsep dual-engine lexicon + IndoBERT. Hanya ide; kode tidak disalin. |
| [dbogdanm/MarketSentiment](https://github.com/dbogdanm/MarketSentiment) | Berita + Fear & Greed + VIX + LLM lokal | 2 | MIT | Acuan tata letak konteks pasar dan gauge. Integrasi Ollama-nya sengaja **tidak** diikuti. |
| [DarmorGamz/Tickermind](https://github.com/DarmorGamz/Tickermind) | Stock scanner dengan LLM lokal | 6 | MIT | Acuan konsep scanner offline/local-first dan status model. |
| [StephanAkkerman/fintwit-bot](https://github.com/StephanAkkerman/fintwit-bot) | Bot sentimen Twitter/Reddit/Binance | 164 | MIT | Pola rate-limit, retry/backoff, dan pemisahan adapter per platform. |

## Referensi data dan API

| Repo | Fokus | Bintang | Lisensi | Pelajaran |
|---|---|---:|---|---|
| [gregyjames/stocktwits-sentiment](https://github.com/gregyjames/stocktwits-sentiment) | StockTwits + Keras/TensorFlow | 36 | MIT | Struktur pesan StockTwits, label bullish/bearish, dan evaluasi model. |
| [khmurakami/pystocktwits](https://github.com/khmurakami/pystocktwits) | Wrapper tidak resmi StockTwits | 22 | MIT | Bentuk endpoint dan field respons; berguna untuk adapter StockTwits. |
| [Shubxam/Nifty-500-Live-Sentiment-Analysis](https://github.com/Shubxam/Nifty-500-Live-Sentiment-Analysis) | Dashboard sentimen live NIFTY 500 | 28 | MIT | Ide watchlist, ranking ticker, dan visualisasi sentimen massal. |
| [JDKrasnick/MarketSent](https://github.com/JDKrasnick/MarketSent) | Reddit + FinBERT + Flask/React | 0 | Tidak ada | Pemisahan API dan frontend; tidak dipakai sebagai dependensi. |
| [a-dorgham/TradeMood](https://github.com/a-dorgham/TradeMood) | VADER + BERT + indikator teknikal | 0 | NOASSERTION | Ide menggabungkan sentimen dan konteks teknikal; lisensi harus diperiksa sebelum memakai kode. |

## Referensi model buatan sendiri

| Repo | Fokus | Bintang | Lisensi | Yang diambil |
|---|---|---:|---|---|
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Membangun LLM dari nol, langkah demi langkah | 107.289 | NOASSERTION | Disiplin tokenisasi, split data, evaluasi, dan pelatihan. Bukan dependensi. |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | GPT minimal untuk pelatihan | 64.078 | MIT | Prinsip model kecil, training loop yang dapat diaudit, dan evaluasi holdout. |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | Autograd dan neural net minimal | 18.056 | MIT | Pemahaman backpropagation; SLM-1 memakai forward/backward numpy sendiri. |
| [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | Melatih LLM kecil dari nol | 64.421 | Apache-2.0 | Inspirasi model kecil yang dilatih dan dievaluasi secara lokal. |

**Posisi proyek ini:** FOMO Meter memakai SLM-1, model saraf kecil buatan sendiri
(hashed n-gram → MLP → softmax), dilatih dari dataset kurasi proyek. Ini bukan
wrapper Ollama, OpenAI, HuggingFace, atau model pihak ketiga. Nama FOMO Meter
mengacu pada psikologi kerumunan pasar; kemampuan dan batasan model tetap
ditampilkan secara jujur di UI.

## Referensi bahasa Indonesia

| Repo | Fokus | Bintang | Lisensi | Catatan |
|---|---|---:|---|---|
| [IndoNLP/indonlu](https://github.com/IndoNLP/indonlu) | Benchmark NLP Indonesia, IndoBERT | 659 | Apache-2.0 | Acuan masa depan untuk pemahaman bahasa Indonesia. Tidak dipakai sebagai dependensi saat ini. |
| [intanq/indonesian-financial-domain-lm](https://github.com/intanq/indonesian-financial-domain-lm) | Post-training IndoBERT untuk domain finansial Indonesia | 3 | Apache-2.0 | Referensi penting untuk rencana peningkatan model Bahasa Indonesia. |

## Referensi desktop dan presentasi

| Repo | Fokus | Bintang | Lisensi | Catatan |
|---|---|---:|---|---|
| [zhiyiYo/PyQt-Fluent-Widgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) | Widget desktop Qt bergaya Fluent | 8.141 | GPL-3.0 | Referensi visual saja. Kode GPL tidak disalin ke proyek ini. |
| [r0x0r/pywebview](https://github.com/r0x0r/pywebview) | GUI Python dengan HTML/CSS/JS | 6.137 | BSD-3-Clause | Alternatif pembungkus web UI; proyek ini memakai PySide6 + QtWebEngine. |

## Keputusan desain yang diambil

1. **Status sumber harus jujur.** Sumber yang gagal atau butuh autentikasi
   ditampilkan sebagai `ERROR`, `NO_DATA`, atau `NEEDS_AUTH`; tidak diisi data
   karangan.
2. **Bukti dipisahkan dari inferensi.** Kutipan berita/post tetap verbatim dan
   memiliki URL; skor model tidak pernah menggantikan bukti.
3. **Relevansi ticker adalah gate wajib.** Post yang hanya menyebut ticker lain
   tidak boleh masuk konsensus.
4. **Dual engine.** Lexicon deterministik menjadi baseline dan fallback;
   SLM-1 digunakan saat tersedia dan cukup yakin.
5. **Rate limit dan cache.** Adapter sumber memakai timeout, retry terbatas, dan
   cache agar tidak membanjiri API publik.
6. **Tanpa Ollama / API model pihak ketiga.** Semua inferensi model terjadi
   lokal dari bobot yang dilatih proyek sendiri.
7. **Konteks pasar bukan sinyal utama.** Fear & Greed, harga, dan volume
   ditampilkan sebagai konteks; tidak boleh mengubah arah sentimen secara diam-diam.

## Yang sengaja tidak dipakai

- Integrasi Ollama dari `MarketSentiment` atau proyek serupa.
- Bobot FinBERT/IndoBERT siap pakai, karena permintaan proyek adalah model
  buatan sendiri. Referensi ini hanya menjadi pembanding dan arah riset.
- Kode UI GPL-3.0 dari PyQt-Fluent-Widgets.
- Scraping tanpa izin atau data sintetis yang dipresentasikan sebagai data live.

## Catatan lisensi

MIT dan Apache-2.0 umumnya permisif, tetapi tetap wajib mencantumkan atribusi
jika kode dipakai. GPL-3.0 bersifat copyleft. Repo tanpa lisensi atau dengan
status `NOASSERTION` harus diperlakukan sebagai **referensi ide saja** sampai
lisensinya diklarifikasi. Proyek ini tidak menyalin kode dari repo referensi.

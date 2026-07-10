# موتور تشخیص سرقت ادبی و اسناد مشابه (Semantic Plagiarism Engine)

یک ابزار خط فرمان (CLI) برای تشخیص سرقت ادبی و اسناد نزدیک به تکراری، پروژه سوم درس داده‌کاوی.

این پروژه دو خط‌لوله (pipeline) تشخیص شباهت اسناد را پیاده‌سازی و با یکدیگر مقایسه می‌کند:

1. Shingling + MinHash + LSH
2. TF-IDF Weighted SimHash

سامانه به‌گونه‌ای طراحی شده که یک ابزار خط فرمان قابل بازتولید باشد، نه یک پروژه صرفاً نوت‌بوکی.

---

## ساختار پروژه

    semantic-plagiarism-engine/
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── data/
    │   ├── sample_corpus/
    │   ├── raw/
    │   └── processed/
    ├── docs/
    │   ├── project_spec.md   
    │   └── project_spec.pdf    (خروجی نهایی PDF)
    ├── notebooks/
    │   └── exploration.ipynb
    ├── outputs/
    ├── src/
    │   └── plagiarism_engine/
    │       ├── preprocessing.py
    │       ├── minhash.py
    │       ├── lsh.py
    │       ├── simhash.py
    │       ├── dataset.py
    │       ├── evaluation.py
    │       └── cli.py
    └── tests/

---

## روش‌های پیاده‌سازی‌شده

### ۱. پیش‌پردازش متن

ماژول پیش‌پردازش عملیات زیر را انجام می‌دهد:

- یکسان‌سازی متن (normalization)
- یکسان‌سازی حروف فارسی/عربی
- حذف نشانه‌گذاری و فاصله‌های اضافه
- حذف URL و ایمیل
- توکن‌سازی در سطح کلمه
- حذف واژه‌های توقف (stopwords)
- شینگل‌سازی کلمه‌ای (word shingling)

اندازه پیش‌فرض شینگل ۳ کلمه است.

### ۲. شباهت جاکارد دقیق (Exact Jaccard)

هر سند به‌صورت مجموعه‌ای از شینگل‌های کلمه‌ای نمایش داده می‌شود.

شباهت جاکارد به‌صورت زیر محاسبه می‌شود:

    J(A, B) = |A ∩ B| / |A ∪ B|

این روش دقیق است اما برای مجموعه اسناد بزرگ، به دلیل نیاز به مقایسه همه جفت‌ها، پرهزینه است.

### ۳. مین‌هش (MinHash)

MinHash برای مجموعه شینگل هر سند، یک امضای فشرده تولید می‌کند.

شباهت تخمینی بین دو سند از نسبت موقعیت‌های برابر در امضاهای آن‌ها محاسبه می‌شود.

طول پیش‌فرض امضا:

    num_perm = 128

### ۴. لوکالیتی سنسیتیو هشینگ (LSH)

LSH امضای MinHash را به چند باند (band) تقسیم می‌کند.

اسنادی که حداقل در یک باکت (bucket) از یک باند مشترک باشند، به‌عنوان جفت کاندید در نظر گرفته می‌شوند.

این روش تعداد مقایسه‌ها را نسبت به مقایسه همه‌به‌همه کاهش می‌دهد.

تنظیم پیش‌فرض:

    num_bands = 32

### ۵. سیم‌هش وزن‌دار با TF-IDF

SimHash برای هر سند یک اثرانگشت (fingerprint) ۶۴ بیتی تولید می‌کند.

هر توکن با استفاده از وزن TF-IDF خود، در یک بردار بیتی وزن‌دار سهیم می‌شود.

شباهت از روی فاصله همینگ محاسبه می‌شود:

    similarity = 1 - hamming_distance / hash_bits

تنظیم پیش‌فرض:

    hash_bits = 64

### ویژگی‌های امتیازی (اختیاری، `src/plagiarism_engine/bonus.py`)

سه ایده تکمیلی که رفتار سه دستور الزامی CLI را تغییر نمی‌دهند و به‌طور مستقل در
`tests/test_bonus.py` آزموده شده‌اند:

- **تنظیم خودکار پارامترهای LSH** (`find_adaptive_lsh_params`): به‌جای انتخاب دستی
  `num_bands`، جفت (تعداد باند، سطر در هر باند) که به یک آستانه شباهت هدف نزدیک‌ترین
  باشد را جست‌وجو می‌کند.
- **بن‌واژه‌ساز سبک فارسی** (`persian_lemmatize`): پسوندهای جمع رایج (`ها`, `های`, `ان`)
  و پیشوندهای فعل استمراری (`می‌`, `نمی‌`) را حذف می‌کند تا صورت‌های صرفی یک واژه در
  شینگل‌سازی یکسان‌تر دیده شوند.
- **سیم‌هش ترکیبی** (`hybrid_simhash`): علاوه بر توکن‌های کلمه‌ای، n-gramهای کاراکتری هر
  کلمه را نیز در محاسبه اثرانگشت دخیل می‌کند تا در برابر غلط‌های تایپی مقاوم‌تر باشد.

این قابلیت‌ها از طریق دستور اختیاری چهارم CLI در دسترس‌اند:

    python -m plagiarism_engine.cli bonus-eval \
      --pairs data/sample_corpus/sample_pairs.csv \
      --text-col-a text_a \
      --text-col-b text_b \
      --label-col label \
      --threshold 0.1 \
      --simhash-threshold 0.5 \
      --output outputs/bonus_metrics.csv

خروجی، معیارهای SimHash استاندارد و SimHash ترکیبی را کنار هم مقایسه می‌کند و پیشنهاد
پارامتر تطبیقی LSH را در خروجی متنی چاپ می‌کند. نمونه کامل و تفسیرشده این قابلیت‌ها در
`notebooks/exploration.ipynb` (بخش «امتیازی») نیز موجود است.

---

## نصب

ساخت محیط مجازی:

    python -m venv .venv

فعال‌سازی در Windows Git Bash:

    source .venv/Scripts/activate

نصب وابستگی‌ها:

    pip install -r requirements.txt
    pip install -e .

توجه: گزارش فنی به‌صورت `docs/project_spec.tex` (LaTeX) نگهداری می‌شود و با
دستور `python scripts/build_report_pdf.py` از روی `docs/project_spec.md`
بازتولید و با XeLaTeX کامپایل می‌شود تا `docs/project_spec.pdf` ساخته شود. برای
اجرای این اسکریپت به یک نصب TeX Live (یا معادل آن) با `xelatex`، بسته‌های
`fontspec` و `babel` نیاز است. فونت فارسی مورد استفاده (Vazirmatn، تحت مجوز
SIL OFL) در مسیر `assets/fonts/` همراه با پروژه قرار داده شده و نیازی به نصب
جداگانه فونت نیست؛ فونت انگلیسی/کد (DejaVu Sans Mono) معمولاً همراه هر نصب
TeX Live یا لینوکس استاندارد موجود است.

---

## اجرای تست‌ها

اجرای همه تست‌ها:

    pytest tests

---

## نحوه استفاده از CLI

### ۱. مقایسه دو سند

    python -m plagiarism_engine.cli compare \
      --file-a data/sample_corpus/doc_01.txt \
      --file-b data/sample_corpus/doc_02.txt \
      --shingle-size 3 \
      --output outputs/two_file_compare.json

خروجی:

    outputs/two_file_compare.json

این خروجی شامل موارد زیر است:

- شباهت جاکارد دقیق
- شباهت MinHash
- اثرانگشت‌های SimHash
- فاصله همینگ SimHash
- شباهت SimHash

---

### ۲. جست‌وجوی اسناد مشابه در یک مجموعه (Corpus)

    python -m plagiarism_engine.cli corpus \
      --data data/sample_corpus \
      --threshold 0.1 \
      --shingle-size 3 \
      --output outputs/candidates.csv

خروجی:

    outputs/candidates.csv

این خروجی شامل جفت‌اسناد کاندید و آمار کاهش محاسبات LSH است.

---

### ۳. ارزیابی روی یک دیتاست جفتی برچسب‌دار

نمونه برای دیتاستی مشابه Quora:

    python -m plagiarism_engine.cli pairs \
      --pairs data/raw/quora/train.csv \
      --text-col-a question1 \
      --text-col-b question2 \
      --label-col is_duplicate \
      --limit 5000 \
      --threshold 0.25 \
      --simhash-threshold 0.75 \
      --output outputs/metrics.csv

خروجی:

    outputs/metrics.csv

این خروجی شامل موارد زیر است:

- accuracy (دقت کلی)
- precision (دقت)
- recall (بازیابی)
- F1-score
- زمان اجرا (runtime)

برای هر سه روش:

- جاکارد دقیق
- MinHash
- SimHash

---

## مجموعه نمونه (Sample Corpus)

مخزن شامل یک مجموعه سند نمونه کوچک است:

    data/sample_corpus/doc_01.txt
    data/sample_corpus/doc_02.txt
    data/sample_corpus/doc_03.txt

این فایل‌ها برای آزمایش سریع CLI استفاده می‌شوند.

---

## سیاست داده (Data Policy)

دیتاست‌های خام بزرگ نباید در GitHub قرار داده شوند.

پوشه‌های زیر توسط Git نادیده گرفته می‌شوند:

    data/raw/
    data/processed/

فقط فایل‌های نمونه کوچک و خروجی‌های نهایی قابل بازتولید باید commit شوند.

---

## وضعیت فعلی

پیاده‌سازی‌شده:

- پیش‌پردازش متن
- شینگل‌سازی کلمه‌ای
- شباهت جاکارد دقیق
- MinHash از صفر
- تولید کاندید با LSH
- SimHash وزن‌دار با TF-IDF
- معیارهای ارزیابی
- ابزارهای بارگذاری دیتاست
- دستورهای CLI
- اجرای آزمایش روی دیتاست واقعی Quora Question Pairs (۵۰۰۰ جفت برچسب‌دار)
- نسخه نهایی `outputs/metrics.csv` و `outputs/pair_predictions.csv` بر پایه نتایج واقعی
- گزارش فنی نهایی (`docs/project_spec.md` / `.tex` / `.pdf`) شامل تحلیل خطا با نمونه‌های واقعی

باقی‌مانده:

- ضبط ویدئوی کوتاه از اجرای CLI
- اطمینان از دسترسی استاد و دستیار آموزشی به مخزن GitHub

---

## خروجی تحلیل خطای سطح جفت (Pair-Level Error Analysis)

دستور `pairs` می‌تواند جزئیات پیش‌بینی سطح جفت را نیز برای تحلیل خطا ذخیره کند.

نمونه:

    python -m plagiarism_engine.cli pairs \
      --pairs data/sample_corpus/sample_pairs.csv \
      --text-col-a text_a \
      --text-col-b text_b \
      --label-col label \
      --threshold 0.1 \
      --simhash-threshold 0.65 \
      --shingle-size 2 \
      --output outputs/metrics.csv \
      --details-output outputs/pair_predictions.csv

فایل جزئیات شامل یک ردیف به‌ازای هر جفت متن است، از جمله:

- برچسب واقعی
- شباهت جاکارد
- شباهت MinHash
- شباهت SimHash
- پیش‌بینی هر روش
- پرچم خطای هر روش

این فایل برای تحلیل موارد مثبت کاذب (false positive) و منفی کاذب (false negative) کاربردی است.

---

## اجرای دمو به‌صورت بازتولیدپذیر

یک اسکریپت دمو برای بازتولید خروجی‌های اصلی ارائه شده است:

    bash scripts/run_demo.sh

این اسکریپت مراحل زیر را اجرا می‌کند:

1. مقایسه دو سند
2. جست‌وجوی اسناد مشابه در سطح مجموعه
3. ارزیابی روی جفت‌های برچسب‌دار نمونه (smoke test، نه ارزیابی واقعی)
4. تولید گزارش PDF

فایل‌های تولیدشده:

    outputs/two_file_compare.json
    outputs/candidates.csv
    outputs/demo_metrics.csv
    outputs/demo_pair_predictions.csv
    docs/project_spec.pdf

توجه: این اسکریپت عمداً `outputs/metrics.csv` و `outputs/pair_predictions.csv` را
بازتولید نمی‌کند. این دو فایل نتایج واقعی ارزیابی روی ۵۰۰۰ جفت Quora Question
Pairs هستند (بخش ۱۱.۲ و ۱۲ گزارش فنی) و چون داده خام آن‌ها (`data/raw/quora/train.csv`)
طبق سیاست پروژه commit نمی‌شود، این خروجی‌ها به‌صورت جداگانه و از پیش تولید و
commit شده‌اند تا با گزارش فنی هم‌خوان بمانند.

---

## چک‌لیست تحویل نهایی

تحویل نهایی شامل موارد زیر است:

- لینک مخزن GitHub
- کد قابل اجرا از طریق CLI
- گزارش فنی PDF
- فایل خروجی معیارها (metrics)
- فایل خروجی جفت‌اسناد کاندید (candidates)
- فایل جزئیات پیش‌بینی سطح جفت برای تحلیل خطا
- ویدئوی کوتاه از اجرای CLI

فایل‌های مهم:

    README.md
    docs/project_spec.pdf
    outputs/metrics.csv
    outputs/candidates.csv
    outputs/pair_predictions.csv
    scripts/run_demo.sh
    notebooks/exploration.ipynb

---

## سناریوی پیشنهادی برای ویدئوی دمو

برای ویدئوی کوتاه، دستورهای زیر را اجرا کنید:

    pytest tests

    bash scripts/run_demo.sh

    cat outputs/candidates.csv

    cat outputs/demo_metrics.csv

    cat outputs/metrics.csv

    head -5 outputs/pair_predictions.csv

مرحله آخر (`cat outputs/metrics.csv`) نتایج واقعی ارزیابی روی ۵۰۰۰ جفت Quora
Question Pairs را نشان می‌دهد (تولیدشده از قبل، چون `run_demo.sh` آن را دوباره
نمی‌سازد)؛ `outputs/demo_metrics.csv` فقط خروجی smoke test روی دیتاست نمونه
کوچک است. ویدئو فقط باید نشان دهد که دستورهای CLI با موفقیت اجرا می‌شوند و
خروجی‌های لازم را تولید می‌کنند؛ نیازی به توضیح کلامی نیست.

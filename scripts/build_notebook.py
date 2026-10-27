"""Generate the standalone, Vietnamese IMDB assignment notebook and initial report."""
from pathlib import Path
from textwrap import dedent
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def code(source, tags=None):
    source = dedent(source).strip()
    ast.parse(source)
    cells.append(nbf.v4.new_code_cell(source, metadata={"tags": tags or []}))


REPORT_BASE = dedent(r"""
# Phân tích cảm xúc IMDB bằng RNN và LSTM

## 1. Mục tiêu và dữ liệu

Phân loại một đánh giá phim tiếng Anh thành negative (0) hoặc positive (1),
so sánh SimpleRNN, LSTM với hai phương pháp truyền thống Logistic Regression
và SVM tuyến tính. Các mô hình được huấn luyện lại trên cùng cách chia dữ liệu.

IMDB Large Movie Review Dataset gồm 50.000 đánh giá có nhãn, cân bằng hai lớp.
Giữ nguyên 25.000 mẫu test chính thức; tách phân tầng 25.000 mẫu train thành
20.000 train và 5.000 validation với seed 42. Không dùng 50.000 mẫu không nhãn.
Ưu tiên tải bản văn bản gốc Stanford; nếu máy chủ không truy cập được, dùng
bản Parquet từ kho `stanfordnlp/imdb`, giữ hai split train/test có sẵn.
CSV không có thông tin split trong thư mục `archive/` không được dùng để tự
chia lại test. ID trong bản Parquet là ID dòng của nguồn, không phải ID phim.

## 2. Cơ sở lý thuyết

### TF-IDF, Logistic Regression và SVM

TF-IDF kết hợp tần suất từ trong tài liệu với độ hiếm trong tập train.
Với `sublinear_tf=True`, TF của từ xuất hiện được biến đổi thành
$1+\log(tf)$. IDF mặc định có làm trơn:
$\mathrm{idf}(t)=\log((1+N)/(1+df(t)))+1$; vector được chuẩn hóa L2.
Unigram biểu diễn từng từ, bigram biểu diễn cặp như “not good”. Cách biểu diễn
này không mô hình hóa đầy đủ thứ tự câu nhưng thường là baseline mạnh.

Logistic Regression dự đoán $p(y=1|x)=\sigma(w^Tx+b)$, tối ưu log loss với
regularization L2. Linear SVM tìm biên phân tách có margin lớn; `LinearSVC`
dùng squared hinge loss và regularization L2 mặc định. Tham số $C$ điều khiển
mức phạt lỗi: $C$ lớn tương ứng regularization yếu hơn. Điểm `decision_function`
của SVM là khoảng cách có dấu tương đối, **không phải xác suất**.

### RNN

Với embedding $x_t$, RNN cập nhật trạng thái:

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b_h),\qquad
\hat y=\sigma(w_o^Th_T+b_o).$$

Trạng thái cuối tổng hợp thông tin theo thứ tự từ. Khi lan truyền ngược qua
nhiều bước thời gian, tích nhiều đạo hàm có thể làm gradient tiêu biến hoặc
bùng nổ. RNN vì vậy khó giữ thông tin ở đầu một đánh giá dài. Gradient clipping
giúp kiểm soát gradient bùng nổ, không giải quyết hoàn toàn gradient tiêu biến.

### LSTM

LSTM bổ sung trạng thái ô nhớ $c_t$ và ba cổng:

$$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),\quad
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i),$$
$$\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c),\quad
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,$$
$$o_t=\sigma(W_o[x_t,h_{t-1}]+b_o),\quad
h_t=o_t\odot\tanh(c_t).$$

Cổng quên chọn thông tin giữ lại, cổng vào chọn thông tin mới, cổng ra điều khiển
trạng thái ẩn. Đường cập nhật cộng của ô nhớ hỗ trợ truyền thông tin xa hơn.
LSTM có nhiều tham số hơn SimpleRNN cùng số units; kết quả thực tế còn phụ thuộc
dữ liệu, regularization và ngân sách huấn luyện, không bảo đảm luôn tốt hơn.

## 3. Thiết kế thí nghiệm

- Làm sạch HTML, giải mã HTML entities, chuyển chữ thường, mở rộng phủ định
  như “isn't” → “is not”, thay dấu câu bằng khoảng trắng. Không bỏ stopword.
- Chỉ `fit` TF-IDF và `adapt` từ vựng trên train. Validation và test chỉ transform.
- Baseline: TF-IDF unigram/bigram, tối đa 20.000 đặc trưng, `min_df=2`,
  `sublinear_tf=True`. Logistic Regression và LinearSVC thử $C\in\{0.1,1,10\}$;
  chọn F1 validation cao nhất, hòa thì chọn C nhỏ hơn. Không refit trên train+val.
- Mạng: Embedding 128 → SimpleRNN/LSTM 64 → Dropout 0,5 → Dense sigmoid.
  Từ vựng chung tối đa 20.000 token (gồm padding/OOV), chuỗi 300 token,
  cắt phần cuối, padding cuối, `mask_zero=True`. Embedding học từ đầu.
- Adam learning rate 0,001, `clipnorm=1.0`, binary cross-entropy, batch 128,
  tối đa 10 epoch. Early stopping theo validation loss, patience 2,
  khôi phục trọng số tốt nhất. Hai mạng đặt lại cùng seed 42 trước khi tạo.
- LSTM dùng tanh/sigmoid, `recurrent_dropout=0`, không unroll, có bias và
  padding cuối để tương thích cuDNN khi GPU/runtime hỗ trợ.
- Dùng Accuracy, Precision, Recall, F1 (positive là lớp 1), ROC-AUC và confusion
  matrix. Ngưỡng dự đoán là 0,5 cho sigmoid/Logistic Regression, 0 cho SVM.
  ROC-AUC dùng xác suất hoặc margin của SVM, không dùng nhãn cứng.
- Chỉ tính kết quả test sau khi chọn C và trọng số mạng bằng validation.
  Theo dõi loss/accuracy train và validation để thảo luận overfitting.

TF-IDF dùng toàn bộ đánh giá, mạng chỉ dùng 300 token đầu. Đây là so sánh các
pipeline thực tế, chưa cô lập riêng ảnh hưởng của kiến trúc. Thời gian fit không
gồm tiền xử lý và đánh giá; baseline cộng thời gian cả ba giá trị C, mạng cộng
các epoch đã chạy. Thời gian tạo đặc trưng được ghi riêng. Baseline chạy CPU;
mạng dùng GPU khi có. So sánh thời gian phụ thuộc phần cứng, không chứng minh
một thuật toán có độ phức tạp thấp hơn. Một seed chưa đo được độ biến thiên
giữa các lần chạy và không đủ để khẳng định ý nghĩa thống kê.

## 4. Kết quả thực nghiệm
""").strip()

REFERENCES = dedent("""
## Tài liệu tham khảo

1. Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL.
   [Bài báo](https://aclanthology.org/P11-1015/),
   [dữ liệu Stanford](https://ai.stanford.edu/~amaas/data/sentiment/).
2. [Bản dữ liệu Stanford NLP trên Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb).
3. [Keras: Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/).
4. [Keras SimpleRNN](https://keras.io/api/layers/recurrent_layers/simple_rnn/),
   [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/).
5. [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
   [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
   [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
""").strip()

md(r"""
    # RNN & LSTM for sentiment analysis — IMDB

    **Bài tập:** so sánh SimpleRNN, LSTM, SVM và Logistic Regression trên 50.000
    đánh giá IMDB. Giải thích, cấu hình và báo cáo bằng tiếng Việt.

    **Cách chạy Colab:** tải notebook lên https://colab.research.google.com/ →
    **Runtime → Change runtime type → GPU (T4 hoặc GPU có sẵn)** → **Run all**.
    Kiểm tra dòng `GPU` ở phần môi trường. Notebook tự tải dữ liệu công khai,
    không cần tài khoản Kaggle hay file Python đi kèm. Nếu mất runtime,
    cần chạy lại hoặc lưu/tải thư mục kết quả trước đó.

    Mặc định `RUN_MODE = "full"`. Chế độ `"quick"` chỉ kiểm tra pipeline với
    1.000 train / 200 validation / 1.000 test và 1 epoch; không dùng để kết luận.
    Các số đo chỉ xuất hiện sau khi thực sự chạy. Thời gian huấn luyện phụ thuộc
    GPU và số epoch early stopping; SimpleRNN vẫn có thể chạy chậm trên GPU.

    Ô cuối tạo file ZIP và nút tải xuống, gồm báo cáo, bảng CSV, hình PNG,
    mô hình đã huấn luyện và thông tin tái lập thí nghiệm.
""")
md("## 1. Cấu hình\nChỉ cần sửa `RUN_MODE` ở ô này để đổi chế độ. Giữ `full` cho bài nộp.")
code('''
    RUN_MODE = "full"  # "full" hoặc "quick"
    SEED = 42
    MAX_TOKENS = 20_000
    SEQUENCE_LENGTH = 300
    EMBEDDING_DIM = 128
    RECURRENT_UNITS = 64
    DROPOUT = 0.5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    C_VALUES = [0.1, 1.0, 10.0]
    DOWNLOAD_RESULTS = False  # True để tự tải ZIP khi chạy trên Colab
    assert RUN_MODE in {"full", "quick"}
    MAX_EPOCHS = 10 if RUN_MODE == "full" else 1
''', ["parameters"])

md("## 2. Thư viện và môi trường\nColab thường có sẵn các thư viện. Nếu thiếu, chạy `%pip install tensorflow scikit-learn pandas numpy matplotlib pyarrow joblib` trong một ô mới rồi khởi động lại runtime.")
code('''
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import sys, re, html, json, tarfile, hashlib, shutil, platform, time, urllib.request
    from pathlib import Path, PurePosixPath
    from datetime import datetime, timezone
    import importlib.metadata
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, ConfusionMatrixDisplay)
    from IPython.display import display, Markdown, FileLink

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    # Giới hạn số luồng CPU cho kiểm tra nhỏ trên máy cá nhân.
    if RUN_MODE == "quick":
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    PROJECT_DIR = Path.cwd()
    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "results" / RUN_MODE
    FIGURE_DIR = OUTPUT_DIR / "figures"
    MODEL_DIR = OUTPUT_DIR / "models"
    for directory in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [tf.config.experimental.get_device_details(g).get("device_name", g.name)
                 for g in gpus]
    versions = {name: importlib.metadata.version(name) for name in
                ["tensorflow", "keras", "scikit-learn", "numpy", "pandas",
                 "matplotlib", "pyarrow", "joblib"]}
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": RUN_MODE, "python": sys.version, "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "gpu": gpu_names, "baseline_device": "CPU",
        "neural_device": "GPU" if gpus else "CPU", "versions": versions,
        "deterministic_ops": True,
    }
    print("Chế độ:", RUN_MODE.upper(), "| GPU:", gpu_names or "không có — dùng CPU")
    display(pd.Series(versions, name="Phiên bản").to_frame())
    if RUN_MODE == "quick":
        display(Markdown("**KIỂM TRA NHANH — kết quả không đại diện cho toàn bộ IMDB.**"))
''')

md("## 3. Tải dữ liệu và giữ nguyên split chính thức\nƯu tiên file Stanford `.tar.gz` đã tải hoặc tải từ máy chủ gốc. Nếu kết nối thất bại, dùng hai split Parquet từ kho Stanford NLP trên Hugging Face. Không dùng phần unsupervised và không tự chia lại test từ CSV.")
code('''
    STANFORD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    HF_BASE = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/"

    def download_file(url, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        print("Đang tải:", url, flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    def sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def read_stanford_archive(archive_path):
        rows = []
        # Đọc trực tiếp văn bản, không giải nén đường dẫn ra filesystem.
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if (not member.isfile() or len(parts) != 4 or parts[0] != "aclImdb"
                        or parts[1] not in {"train", "test"}
                        or parts[2] not in {"pos", "neg"} or not parts[3].endswith(".txt")):
                    continue
                with archive.extractfile(member) as stream:
                    text = stream.read().decode("utf-8")
                rows.append({"review_id": "/".join(parts[1:]), "official_split": parts[1],
                             "text": text, "label": int(parts[2] == "pos")})
        return pd.DataFrame(rows).sort_values("review_id").reset_index(drop=True)

    def load_imdb():
        archive = DATA_DIR / "aclImdb_v1.tar.gz"
        mirror_files = {s: DATA_DIR / "stanfordnlp_imdb" / f"{s}.parquet"
                        for s in ["train", "test"]}
        if not archive.exists() and not all(p.exists() for p in mirror_files.values()):
            try:
                download_file(STANFORD_URL, archive)
            except Exception as exc:
                print("Máy chủ Stanford chưa truy cập được:", type(exc).__name__)
                print("Chuyển sang bản Stanford NLP giữ split train/test.")
        if archive.exists():
            frame = read_stanford_archive(archive)
            provenance = {"source": "Stanford raw archive", "urls": [STANFORD_URL],
                          "sha256": {archive.name: sha256_file(archive)}}
        else:
            frames, urls = [], []
            for split, path in mirror_files.items():
                url = HF_BASE + f"{split}-00000-of-00001.parquet"
                urls.append(url)
                if not path.exists():
                    download_file(url, path)
                part = pd.read_parquet(path)[["text", "label"]].reset_index(drop=True)
                part["review_id"] = [f"{split}/{i:05d}" for i in range(len(part))]
                part["official_split"] = split
                frames.append(part)
            frame = pd.concat(frames, ignore_index=True)
            provenance = {"source": "stanfordnlp/imdb Parquet (official splits)",
                          "urls": urls,
                          "sha256": {s: sha256_file(p) for s, p in mirror_files.items()}}
        assert len(frame) == 50_000 and frame.review_id.is_unique
        assert frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
        assert set(frame.label.unique()) == {0, 1}
        for split in ["train", "test"]:
            counts = frame.loc[frame.official_split.eq(split), "label"].value_counts().to_dict()
            assert counts == {0: 12_500, 1: 12_500}, (split, counts)
        return frame, provenance

    raw_df, provenance = load_imdb()
    environment["dataset"] = provenance
    print("Nguồn:", provenance["source"])
    display(pd.crosstab(raw_df.official_split, raw_df.label))
''')

md("## 4. Chia train/validation, làm sạch và kiểm tra dữ liệu\nTách validation trước mọi bước học từ vựng. Quick chỉ lấy mẫu phân tầng từ ba tập đã tách, giữ nguyên ranh giới train/test chính thức.")
code(r'''
    def clean_text(text):
        text = html.unescape(str(text)).lower().replace("’", "'")
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"<[^>]*>", " ", text)
        for contraction, expanded in [("can't", "can not"), ("won't", "will not"),
                                       ("shan't", "shall not")]:
            text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
        text = re.sub(r"n\'t\b", " not", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    assert clean_text("<br />I can't say it isn't good! &amp;") == "i can not say it is not good"
    official_train = raw_df[raw_df.official_split.eq("train")].copy()
    official_test = raw_df[raw_df.official_split.eq("test")].copy()
    train_df, val_df = train_test_split(official_train, test_size=0.2,
                                       stratify=official_train.label, random_state=SEED)
    test_df = official_test.copy()

    def stratified_subset(frame, n):
        if len(frame) <= n:
            return frame.copy()
        subset, _ = train_test_split(frame, train_size=n, stratify=frame.label,
                                     random_state=SEED)
        return subset.copy()

    if RUN_MODE == "quick":
        train_df = stratified_subset(train_df, 1000)
        val_df = stratified_subset(val_df, 200)
        test_df = stratified_subset(test_df, 1000)
    splits = {"train": train_df, "validation": val_df, "test": test_df}
    expected_sizes = [20_000, 5_000, 25_000] if RUN_MODE == "full" else [1000, 200, 1000]
    for (split, frame), expected in zip(splits.items(), expected_sizes):
        frame.reset_index(drop=True, inplace=True)
        frame["clean_text"] = frame.text.map(clean_text)
        frame["n_tokens"] = frame.clean_text.str.split().str.len()
        assert len(frame) == expected
        assert frame.label.value_counts().to_dict() == {0: expected // 2, 1: expected // 2}
        assert frame.n_tokens.gt(0).all()
    id_sets = [set(frame.review_id) for frame in splits.values()]
    assert all(id_sets[i].isdisjoint(id_sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert train_df.official_split.eq("train").all() and val_df.official_split.eq("train").all()
    assert test_df.official_split.eq("test").all()
    manifest = pd.concat([frame[["review_id", "official_split", "label"]].assign(split=split)
                          for split, frame in splits.items()], ignore_index=True)
    manifest.to_csv(OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    split_summary = pd.DataFrame([
        {"split": split, "samples": len(frame), "negative": int(frame.label.eq(0).sum()),
         "positive": int(frame.label.eq(1).sum()), "mean_tokens": frame.n_tokens.mean(),
         "p95_tokens": frame.n_tokens.quantile(0.95),
         "truncated_fraction": frame.n_tokens.gt(SEQUENCE_LENGTH).mean()}
        for split, frame in splits.items()])
    display(split_summary.round(4))
    split_summary.to_csv(OUTPUT_DIR / "data_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    split_summary.set_index("split")[["negative", "positive"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Class distribution", ylabel="Reviews")
    axes[1].hist(train_df.n_tokens, bins=60, color="#26828e")
    axes[1].axvline(SEQUENCE_LENGTH, color="crimson", linestyle="--", label="300 tokens")
    axes[1].set(title="Training review lengths", xlabel="Tokens", ylabel="Reviews")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "data_overview.png", dpi=160)
    plt.show()
''')

md("## 5. Cơ sở lý thuyết\n" + REPORT_BASE.split("## 2. Cơ sở lý thuyết\n", 1)[1].split("## 3. Thiết kế thí nghiệm", 1)[0])

md("## 6. Học đặc trưng chỉ từ train\nHai mạng dùng cùng từ vựng. TF-IDF dùng toàn bộ đánh giá; chuỗi số nguyên chỉ giữ 300 token đầu. Ô này còn kiểm tra bằng một từ chỉ xuất hiện ở validation giả lập để phát hiện học nhầm vocabulary.")
code(r'''
    def fit_preprocessors(training_texts):
        tfidf = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b",
                                ngram_range=(1, 2), max_features=MAX_TOKENS,
                                min_df=2, sublinear_tf=True, dtype=np.float32)
        start = time.perf_counter()
        tfidf.fit(training_texts)
        tfidf_fit_seconds = time.perf_counter() - start
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=MAX_TOKENS, standardize=None, split="whitespace",
            output_mode="int", output_sequence_length=SEQUENCE_LENGTH)
        start = time.perf_counter()
        text_ds = tf.data.Dataset.from_tensor_slices(training_texts).batch(BATCH_SIZE)
        vectorizer.adapt(text_ds)
        sequence_fit_seconds = time.perf_counter() - start
        return tfidf, vectorizer, tfidf_fit_seconds, sequence_fit_seconds

    # Gọi cùng hàm fit được dùng cho dữ liệu thật; token của validation/test bị giữ ngoài fit.
    probe_train = ["good movie", "good story", "bad movie", "bad story"]
    probe_val = ["validationonlysentinel good"]
    probe_test = ["testonlysentinel bad"]
    probe_tfidf, probe_vectorizer, _, _ = fit_preprocessors(probe_train)
    for held_out in probe_val + probe_test:
        sentinel = held_out.split()[0]
        assert sentinel not in probe_tfidf.vocabulary_
        assert sentinel not in probe_vectorizer.get_vocabulary()
        assert probe_vectorizer(tf.constant([held_out])).numpy()[0, 0] == 1  # OOV
    del probe_tfidf, probe_vectorizer

    tfidf, vectorizer, tfidf_fit_seconds, sequence_fit_seconds = fit_preprocessors(
        train_df.clean_text.tolist())
    assert vectorizer.get_vocabulary()[0] == ""  # padding = 0
    assert len(vectorizer.get_vocabulary()) <= MAX_TOKENS

    def encode_sequences(texts):
        batches = tf.data.Dataset.from_tensor_slices(list(texts)).batch(BATCH_SIZE)
        return np.concatenate([vectorizer(batch).numpy().astype(np.int32) for batch in batches])

    feature_times, tfidf_features, sequence_features = [], {}, {}
    for split, frame in splits.items():
        texts = frame.clean_text.tolist()
        start = time.perf_counter()
        tfidf_features[split] = tfidf.transform(texts)
        tfidf_transform_seconds = time.perf_counter() - start
        start = time.perf_counter()
        encoded = encode_sequences(texts)
        sequence_features[split] = encoded
        feature_times.append({"split": split, "tfidf_transform_seconds": tfidf_transform_seconds,
                              "sequence_transform_seconds": time.perf_counter() - start})
        assert encoded.shape == (len(frame), SEQUENCE_LENGTH)
        assert encoded.min() >= 0 and encoded.max() < len(vectorizer.get_vocabulary())
        assert not np.any((encoded[:, :-1] == 0) & (encoded[:, 1:] != 0)), "Padding phải ở cuối"
    feature_timing = {"tfidf_fit_seconds": tfidf_fit_seconds,
                      "sequence_vocabulary_fit_seconds": sequence_fit_seconds,
                      "transform": feature_times}
    y_train = train_df.label.to_numpy(dtype=np.int32)
    y_val = val_df.label.to_numpy(dtype=np.int32)
    y_test = test_df.label.to_numpy(dtype=np.int32)
    joblib.dump(tfidf, MODEL_DIR / "tfidf.joblib")
    (MODEL_DIR / "vocabulary.json").write_text(
        json.dumps(vectorizer.get_vocabulary(), ensure_ascii=False), encoding="utf-8")
    display(pd.DataFrame(feature_times).round(3))
    print("TF-IDF dimensions:", tfidf_features["train"].shape[1],
          "| Vocabulary size:", len(vectorizer.get_vocabulary()))
''')

md("## 7. Huấn luyện Logistic Regression và SVM\nThử ba giá trị C theo thứ tự tăng dần, chọn F1 validation tốt nhất; trường hợp hòa giữ C nhỏ hơn. Không đọc nhãn test trong quá trình chọn mô hình. Thời gian train của baseline cộng cả ba lần fit; tiền xử lý được báo riêng.")
code('''
    baseline_models, training_details, candidate_rows = {}, {}, []
    factories = {
        "Logistic Regression": lambda c: LogisticRegression(
            C=c, solver="liblinear", max_iter=2000, random_state=SEED),
        "SVM": lambda c: LinearSVC(C=c, max_iter=5000, random_state=SEED),
    }
    for name, factory in factories.items():
        best_f1, best_model, best_c, best_fit_seconds = -1.0, None, None, None
        total_fit_seconds = 0.0
        for c in sorted(C_VALUES):
            candidate = factory(c)
            start = time.perf_counter()
            candidate.fit(tfidf_features["train"], y_train)
            elapsed = time.perf_counter() - start
            total_fit_seconds += elapsed
            score = f1_score(y_val, candidate.predict(tfidf_features["validation"]), zero_division=0)
            candidate_rows.append({"model": name, "C": c, "validation_f1": score,
                                   "fit_seconds": elapsed, "run_mode": RUN_MODE})
            if score > best_f1:
                best_f1, best_model, best_c, best_fit_seconds = score, candidate, c, elapsed
        baseline_models[name] = best_model
        training_details[name] = {"C": best_c, "validation_f1": best_f1,
                                  "train_seconds": total_fit_seconds,
                                  "selected_fit_seconds": best_fit_seconds, "device": "CPU"}
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUTPUT_DIR / "baseline_validation.csv", index=False, encoding="utf-8-sig")
    joblib.dump(baseline_models, MODEL_DIR / "baseline_classifiers.joblib")
    display(candidates.round(4))
    print("C được chọn:", {name: info["C"] for name, info in training_details.items()})
''')

md("## 8. Huấn luyện RNN và LSTM\nChỉ thay lớp hồi tiếp để so sánh hai mạng. LSTM có nhiều tham số hơn với cùng 64 units. Embedding bỏ qua padding; chuỗi padding cuối hỗ trợ cuDNN cho LSTM. Early stopping và checkpoint đều theo validation loss; test chưa được đánh giá.")
code('''
    def build_neural_model(kind):
        recurrent = tf.keras.layers.SimpleRNN if kind == "RNN" else tf.keras.layers.LSTM
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32"),
            tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), EMBEDDING_DIM,
                                      mask_zero=True, name="embedding"),
            recurrent(RECURRENT_UNITS, activation="tanh", recurrent_dropout=0.0,
                      name="recurrent"),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ], name=kind)
        model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=LEARNING_RATE, clipnorm=1.0),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def neural_dataset(features, labels, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if training:
            dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    neural_models, histories = {}, {}
    for name in ["RNN", "LSTM"]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = build_neural_model(name)
        model.summary()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                             restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / f"{name}.keras"),
                monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        start = time.perf_counter()
        history = model.fit(
            neural_dataset(sequence_features["train"], y_train, training=True),
            validation_data=neural_dataset(sequence_features["validation"], y_val),
            epochs=MAX_EPOCHS, callbacks=callbacks, verbose=2)
        elapsed = time.perf_counter() - start
        hist = pd.DataFrame(history.history)
        assert np.isfinite(hist.to_numpy()).all(), "Loss/metric không hữu hạn"
        hist.index = np.arange(1, len(hist) + 1)
        hist.index.name = "epoch"
        hist.to_csv(OUTPUT_DIR / f"history_{name}.csv", encoding="utf-8-sig")
        best_epoch = int(hist.val_loss.idxmin())
        # Load checkpoint chọn bằng validation: khớp trọng số được dùng để đánh giá và xuất.
        best_model = tf.keras.models.load_model(MODEL_DIR / f"{name}.keras")
        neural_models[name], histories[name] = best_model, hist
        training_details[name] = {
            "train_seconds": elapsed, "epochs_run": len(hist), "best_epoch": best_epoch,
            "best_validation_loss": float(hist.loc[best_epoch, "val_loss"]),
            "best_validation_accuracy": float(hist.loc[best_epoch, "val_accuracy"]),
            "parameters": int(best_model.count_params()),
            "device": environment["neural_device"],
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (name, hist) in enumerate(histories.items()):
        for col, metric in enumerate(["loss", "accuracy"]):
            ax = axes[row, col]
            ax.plot(hist.index, hist[metric], marker="o", label="train")
            ax.plot(hist.index, hist["val_" + metric], marker="o", label="validation")
            ax.axvline(training_details[name]["best_epoch"], color="gray", linestyle="--",
                       label="best val_loss epoch")
            ax.set(title=f"{name}: {metric}", xlabel="Epoch", ylabel=metric)
            ax.set_xticks(hist.index)
            ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "learning_curves.png", dpi=160)
    plt.show()
''')

md("## 9. Đánh giá bốn mô hình trên cùng test\nChỉ đến đây mới tính dự đoán và chỉ số test. Không dùng kết quả này để chọn lại C, epoch hay ngưỡng. Precision/Recall/F1 tính cho lớp positive; SVM dùng margin để tính ROC-AUC.")
code('''
    model_order = ["Logistic Regression", "SVM", "RNN", "LSTM"]
    metric_rows, test_outputs = [], {}
    predictions_df = test_df[["review_id", "label", "n_tokens"]].copy()
    for name in model_order:
        start = time.perf_counter()
        if name == "SVM":
            scores = baseline_models[name].decision_function(tfidf_features["test"])
            predicted = (scores >= 0).astype(np.int32)
        elif name == "Logistic Regression":
            scores = baseline_models[name].predict_proba(tfidf_features["test"])[:, 1]
            predicted = (scores >= 0.5).astype(np.int32)
        else:
            scores = neural_models[name].predict(sequence_features["test"],
                                                 batch_size=BATCH_SIZE, verbose=0).ravel()
            predicted = (scores >= 0.5).astype(np.int32)
        predict_seconds = time.perf_counter() - start
        assert scores.shape == y_test.shape and np.isfinite(scores).all()
        assert set(np.unique(predicted)).issubset({0, 1})
        test_outputs[name] = {"scores": scores, "predicted": predicted}
        metric_rows.append({
            "model": name, "run_mode": RUN_MODE, "test_samples": len(y_test),
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_test, scores),
            "train_seconds": training_details[name]["train_seconds"],
            "predict_seconds": predict_seconds, "device": training_details[name]["device"],
        })
        slug = name.lower().replace(" ", "_")
        predictions_df[slug + "_prediction"] = predicted
        predictions_df[slug + "_score"] = scores
    metrics = pd.DataFrame(metric_rows).set_index("model").loc[model_order].reset_index()
    assert metrics.shape[0] == 4
    assert metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]].apply(
        lambda col: col.between(0, 1).all()).all()
    assert metrics.test_samples.eq(len(test_df)).all()
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    display(metrics.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    metrics.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(
        ax=axes[0], rot=15, ylim=(0, 1.12))
    axes[0].set(title=f"Test metrics — {RUN_MODE.upper()}", ylabel="Score", xlabel="")
    axes[0].legend(loc="upper center", ncol=5, fontsize=8)
    axes[1].bar(metrics.model, metrics.train_seconds, color=["#355f8d", "#2a788e", "#22a884", "#7ad151"])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(title="Training (includes C search for baselines)", ylabel="Seconds")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "comparison.png", dpi=160)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.flat, model_order):
        ConfusionMatrixDisplay.from_predictions(y_test, test_outputs[name]["predicted"],
            labels=[0, 1], display_labels=["negative", "positive"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name)
    fig.suptitle(f"Confusion matrices — {RUN_MODE.upper()}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrices.png", dpi=160)
    plt.show()
''')

md("## 10. Phân tích lỗi và thử dự đoán\nLấy tối đa ba lỗi có điểm dự đoán xa ngưỡng nhất của mỗi mô hình. Điểm SVM và xác suất không cùng thang đo. Thống kê theo độ dài chỉ mô tả tương quan, chưa chứng minh cắt chuỗi gây lỗi. Phủ định, mỉa mai và cảm xúc trái chiều cần đọc từng ví dụ để giải thích.")
code('''
    error_frames, length_rows = [], []
    for name in model_order:
        output = test_outputs[name]
        wrong = output["predicted"] != y_test
        errors = test_df.loc[wrong, ["review_id", "text", "label", "n_tokens"]].copy()
        errors["predicted"] = output["predicted"][wrong]
        errors["score"] = output["scores"][wrong]
        errors["distance_from_threshold"] = np.abs(errors.score - (0 if name == "SVM" else 0.5))
        errors["model"] = name
        errors["has_negation"] = errors.text.map(clean_text).str.contains(r"\\b(not|no|never)\\b", regex=True)
        errors["truncated_for_neural"] = errors.n_tokens.gt(SEQUENCE_LENGTH)
        error_frames.append(errors.sort_values("distance_from_threshold", ascending=False).head(3))
        for long_review in [False, True]:
            mask = test_df.n_tokens.gt(SEQUENCE_LENGTH).to_numpy() == long_review
            length_rows.append({"model": name, "length_group": ">300" if long_review else "<=300",
                                "samples": int(mask.sum()),
                                "error_rate": float(wrong[mask].mean()) if mask.any() else None})
    error_examples = pd.concat(error_frames, ignore_index=True)
    length_analysis = pd.DataFrame(length_rows)
    error_examples.to_csv(OUTPUT_DIR / "error_examples.csv", index=False, encoding="utf-8-sig")
    length_analysis.to_csv(OUTPUT_DIR / "errors_by_length.csv", index=False, encoding="utf-8-sig")
    display(error_examples.assign(text=error_examples.text.str.slice(0, 240))[
        ["model", "review_id", "label", "predicted", "score", "has_negation", "text"]])
    display(length_analysis.round(4))

    def predict_reviews(texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts or any(not isinstance(text, str) or not clean_text(text) for text in texts):
            raise ValueError("Cần ít nhất một đánh giá có nội dung sau khi làm sạch.")
        cleaned = [clean_text(text) for text in texts]
        sparse = tfidf.transform(cleaned)
        sequences = encode_sequences(cleaned)
        rows = []
        for name in model_order:
            if name == "SVM":
                scores = baseline_models[name].decision_function(sparse)
            elif name == "Logistic Regression":
                scores = baseline_models[name].predict_proba(sparse)[:, 1]
            else:
                scores = neural_models[name].predict(sequences, verbose=0).ravel()
            for text, score in zip(texts, scores):
                rows.append({"text": text, "model": name, "score": float(score),
                             "score_type": "margin" if name == "SVM" else "probability",
                             "sentiment": "positive" if score >= (0 if name == "SVM" else 0.5) else "negative"})
        return pd.DataFrame(rows)

    display(predict_reviews(["The acting was wonderful and I loved the story.",
                             "I did not enjoy this boring movie at all."]))
    assert predict_reviews("A good movie.").shape[0] == 4
    try:
        predict_reviews("<br />!!!")
    except ValueError:
        pass
    else:
        raise AssertionError("Đầu vào trống phải được phát hiện")
    # Kiểm tra file mô hình baseline và vocabulary đủ để phục hồi dự đoán.
    restored_tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    restored_baselines = joblib.load(MODEL_DIR / "baseline_classifiers.joblib")
    fixture = train_df.clean_text.iloc[:3].tolist()
    for name in baseline_models:
        np.testing.assert_array_equal(
            restored_baselines[name].predict(restored_tfidf.transform(fixture)),
            baseline_models[name].predict(tfidf.transform(fixture)))
    restored_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, standardize=None, split="whitespace", output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        vocabulary=json.loads((MODEL_DIR / "vocabulary.json").read_text(encoding="utf-8")))
    np.testing.assert_array_equal(restored_vectorizer(tf.constant(fixture)).numpy(), encode_sequences(fixture))
    assert all((MODEL_DIR / f"{name}.keras").exists() for name in ["RNN", "LSTM"])
    print("Kiểm tra dữ liệu, metric, đầu vào và phục hồi mô hình: PASS")
''')

md("## 11. Xuất báo cáo, cấu hình và ZIP\nChạy full sẽ cập nhật `Bao_cao_IMDB.md` ở thư mục hiện tại. Quick chỉ ghi báo cáo trong `results/quick`, không ghi đè báo cáo đầy đủ. ZIP không chứa dữ liệu gốc; có mô hình và biểu đồ. Trong Colab có thể bật `DOWNLOAD_RESULTS=True` hoặc chạy ô tải thủ công bên dưới.")
code("REPORT_BASE = " + repr(REPORT_BASE) + "\nREFERENCES = " + repr(REFERENCES) + "\n" + dedent(r'''
    config = {
        "run_mode": RUN_MODE, "seed": SEED, "max_tokens": MAX_TOKENS,
        "sequence_length": SEQUENCE_LENGTH, "embedding_dim": EMBEDDING_DIM,
        "recurrent_units": RECURRENT_UNITS, "dropout": DROPOUT, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE, "clipnorm": 1.0,
        "early_stopping": {"monitor": "val_loss", "patience": 2, "restore_best_weights": True},
        "C_values": C_VALUES, "tfidf": {"ngram_range": [1, 2], "min_df": 2,
        "sublinear_tf": True, "max_features": MAX_TOKENS},
        "split_sizes": {split: len(frame) for split, frame in splits.items()},
        "vocabulary_size": len(vectorizer.get_vocabulary()),
        "tfidf_dimensions": len(tfidf.vocabulary_),
        "selection": training_details, "feature_timing": feature_timing,
        "preprocessing": "HTML unescape/strip; lowercase; expand n't; ASCII letters/digits; whitespace",
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "checks.json").write_text(json.dumps({
        "run_mode": RUN_MODE, "official_dataset_counts": "PASS", "stratification": "PASS",
        "split_ids_disjoint": "PASS", "train_only_vocabulary_sentinel": "PASS",
        "right_padding": "PASS", "finite_metrics": "PASS", "four_models": "PASS",
        "reload_baselines": "PASS", "reload_neural_checkpoints": "PASS",
        "reload_vocabulary": "PASS", "empty_input": "PASS",
    }, indent=2), encoding="utf-8")

    def markdown_table(frame):
        def fmt(value):
            if isinstance(value, (float, np.floating)):
                return f"{value:.4f}" if np.isfinite(value) else "N/A"
            return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
                 "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        lines += ["| " + " | ".join(fmt(v) for v in row) + " |"
                  for row in frame.itertuples(index=False, name=None)]
        return "\n".join(lines)

    def make_report(image_prefix="figures"):
        warning = ("> **KIỂM TRA NHANH: 1.000 train / 200 validation / 1.000 test, 1 epoch. "
                   "Không dùng bảng này để kết luận cho thí nghiệm đầy đủ. "
                   "Mục thiết kế bên dưới mô tả cấu hình full.**\n\n") if RUN_MODE == "quick" else ""
        report = warning + REPORT_BASE + "\n\n"
        report += f"Lần chạy: **{RUN_MODE.upper()}**. Nguồn: {provenance['source']}. "
        report += f"Thời điểm UTC: {environment['created_utc']}.\n\n"
        report += "### 4.1. Dữ liệu thực tế và tài nguyên\n\n" + markdown_table(split_summary) + "\n\n"
        report += f"CPU: {environment['cpu']}; GPU: {', '.join(gpu_names) or 'không có'}. "
        report += f"Baseline: CPU; mạng: {environment['neural_device']}. "
        report += f"Vocabulary: {config['vocabulary_size']}; TF-IDF: {config['tfidf_dimensions']} đặc trưng.\n\n"
        report += f"![Phân bố dữ liệu]({image_prefix}/data_overview.png)\n\n"
        report += "### 4.2. Chọn cấu hình bằng validation\n\n" + markdown_table(candidates) + "\n\n"
        selection = pd.DataFrame([dict(model=name, **training_details[name]) for name in model_order]).fillna("")
        report += markdown_table(selection) + "\n\n"
        report += "### 4.3. So sánh trên cùng tập test\n\n" + markdown_table(metrics) + "\n\n"
        report += f"![So sánh mô hình]({image_prefix}/comparison.png)\n\n"
        report += f"![Confusion matrices]({image_prefix}/confusion_matrices.png)\n\n"
        report += "Thời gian tạo đặc trưng (tách khỏi thời gian fit):\n\n"
        report += f"TF-IDF fit: {tfidf_fit_seconds:.3f}s; vocabulary adapt: {sequence_fit_seconds:.3f}s.\n\n"
        report += markdown_table(pd.DataFrame(feature_times)) + "\n\n"
        report += "### 4.4. Học theo epoch và overfitting\n\n"
        report += f"![Learning curves]({image_prefix}/learning_curves.png)\n\n"
        for name, hist in histories.items():
            epoch = training_details[name]["best_epoch"]
            gap = hist.loc[epoch, "accuracy"] - hist.loc[epoch, "val_accuracy"]
            report += (f"- {name}: đã chạy {len(hist)} epoch, chọn epoch {epoch}; "
                       f"train accuracy − validation accuracy tại epoch chọn = {gap:.4f}.\n")
        report += ("\nKhoảng cách train/validation và xu hướng validation loss là dấu hiệu cần "
                   "xem xét cùng nhau. Validation loss tăng trong khi train loss giảm có thể "
                   "gợi ý overfitting. Một epoch trong quick chưa đủ đánh giá xu hướng.\n\n")
        report += "### 4.5. Phân tích lỗi\n\n" + markdown_table(length_analysis) + "\n\n"
        report += ("Bảng theo độ dài mô tả tương quan; độ khó nội dung và độ dài cùng thay đổi "
                   "nên không thể kết luận việc cắt chuỗi là nguyên nhân duy nhất. "
                   "Các ví dụ sau được chọn tự động từ những dự đoán sai xa ngưỡng nhất. "
                   "`has_negation` chỉ đánh dấu có not/no/never, không xác định phạm vi phủ định.\n\n")
        for _, example in error_examples.iterrows():
            excerpt = html.escape(" ".join(example.text.split())[:500])
            report += (f"**{example.model} — {example.review_id}:** nhãn thật {example.label}, "
                       f"dự đoán {example.predicted}, điểm {example.score:.4f}, "
                       f"{example.n_tokens} token, có phủ định: {example.has_negation}.\n\n")
            report += "> " + excerpt + "\n\n"
        report += ("Để diễn giải từng lỗi, đọc cả đánh giá gốc trong `error_examples.csv`: "
                   "phủ định có thể đảo nghĩa của từ tích cực; mỉa mai hoặc cảm xúc trái chiều "
                   "đòi hỏi ngữ cảnh; phần kết luận sau token 300 bị mạng bỏ qua. "
                   "Các cơ chế này là hướng kiểm tra, chưa phải nguyên nhân đã được xác nhận "
                   "cho mọi ví dụ. TF-IDF có thể học bigram nhưng thiếu phụ thuộc dài; "
                   "RNN có thể mất tín hiệu xa; LSTM có thể vẫn overfit khi học embedding từ đầu.\n\n")
        report += "## 5. Nhận xét và kết luận\n\n"
        if RUN_MODE == "quick":
            report += ("Pipeline đã chạy đủ bốn mô hình và xuất bảng/hình/mô hình. "
                       "Cỡ mẫu nhỏ và một epoch chỉ phục vụ kiểm tra kỹ thuật; "
                       "cần chạy FULL trên Colab GPU để kết luận bài tập.\n\n")
        else:
            indexed = metrics.set_index("model")
            best = metrics.sort_values("f1", ascending=False, kind="stable").iloc[0]
            delta = indexed.loc["LSTM", "f1"] - indexed.loc["RNN", "f1"]
            baseline_best = indexed.loc[["Logistic Regression", "SVM"], "f1"].max()
            neural_best = indexed.loc[["RNN", "LSTM"], "f1"].max()
            report += (f"Trong lần chạy này, {best['model']} đạt F1 cao nhất hoặc đồng cao nhất "
                       f"({best.f1:.4f}), accuracy {best.accuracy:.4f}. "
                       f"Chênh lệch F1 LSTM − RNN là {delta:+.4f}; "
                       f"F1 mạng tốt nhất − baseline tốt nhất là {neural_best - baseline_best:+.4f}. "
                       "Các số đo cho biết kết quả của cấu hình và seed hiện tại.\n\n")
            report += ("Ưu thế phải được xem cùng Precision/Recall, chi phí huấn luyện và giới hạn "
                       "cắt chuỗi. TF-IDF có lợi khi từ/cặp từ mang tín hiệu mạnh; LSTM có khả năng "
                       "giữ ngữ cảnh tốt hơn SimpleRNN nhưng khả năng đó không bảo đảm thắng "
                       "baseline. Cần nhiều seed hoặc thí nghiệm kiểm soát độ dài để kiểm tra "
                       "độ ổn định và nguyên nhân khác biệt; không chỉnh cấu hình dựa trên test này.\n\n")
        report += "## 6. Khả năng tái lập\n\n"
        report += ("Cấu hình đầy đủ ở `config.json`; phần cứng, phiên bản, URL và SHA-256 dữ liệu "
                   "ở `environment.json`; ID và nhãn từng split ở `split_manifest.csv`. "
                   "`checks.json` ghi các kiểm tra đã vượt qua. Đã đặt seed và deterministic ops; "
                   "kết quả vẫn có thể khác giữa phiên bản thư viện hoặc phần cứng.\n\n")
        report += REFERENCES + "\n"
        return report

    report_name = "Bao_cao_IMDB.md" if RUN_MODE == "full" else "Bao_cao_IMDB_quick.md"
    report_path = OUTPUT_DIR / report_name
    report_path.write_text(make_report(), encoding="utf-8")
    if RUN_MODE == "full":
        (PROJECT_DIR / "Bao_cao_IMDB.md").write_text(
            make_report("results/full/figures"), encoding="utf-8")
    zip_base = PROJECT_DIR / "results" / f"IMDB_{RUN_MODE}_results"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_DIR))
    print("Báo cáo:", report_path)
    print("ZIP:", zip_path, f"({zip_path.stat().st_size / 1024**2:.1f} MB)")
    display(Markdown("**Hoàn tất.** Các kết quả được ghi nhận từ lần chạy thực tế này."))
    display(FileLink(str(report_path.relative_to(PROJECT_DIR))))
    display(FileLink(str(zip_path.relative_to(PROJECT_DIR))))
    if DOWNLOAD_RESULTS:
        try:
            from google.colab import files
        except ImportError:
            print("Ngoài Colab: tải qua liên kết ZIP ở trên.")
        else:
            files.download(str(zip_path))
'''))

md("## 12. Tải kết quả trên Colab\nSau khi Run all hoàn tất, mở bảng **Files** bên trái, chọn `results/IMDB_full_results.zip` → menu ba chấm → **Download**. Hoặc chạy một ô mới với:\n\n```python\nfrom google.colab import files\nfiles.download(str(zip_path))\n```\n\nTải notebook đã chạy bằng **File → Download → Download .ipynb** để giữ output trong bài nộp. ZIP chỉ chứa kết quả và mô hình, không tự chứa bản notebook đang mở.\n\n" + REFERENCES)


def main():
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "IMDB_Sentiment_Analysis.ipynb", "provenance": []},
        "accelerator": "GPU",
    })
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / "IMDB_Sentiment_Analysis.ipynb")
    report = ROOT / "Bao_cao_IMDB.md"
    if not report.exists():
        pending = (
            "\n\n**Trạng thái: chờ chạy FULL trên Google Colab GPU.**\n\n"
            "Chưa có số liệu huấn luyện đầy đủ. Mở `IMDB_Sentiment_Analysis.ipynb`, "
            "giữ `RUN_MODE = \"full\"`, bật GPU và Run all. Ô xuất báo cáo tự cập nhật "
            "bảng Accuracy/Precision/Recall/F1/ROC-AUC, thời gian, biểu đồ và kết luận. "
            "Không sử dụng kết quả quick để thay thế kết quả full.\n\n"
            "Kết quả kiểm tra nhanh, nếu đã chạy, nằm riêng trong "
            "[báo cáo quick](results/quick/Bao_cao_IMDB_quick.md).\n\n"
        )
        report.write_text(REPORT_BASE + pending + REFERENCES + "\n", encoding="utf-8")
    print(f"Generated {len(cells)} cells; all code cells passed Python syntax validation.")


if __name__ == "__main__":
    main()

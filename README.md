# Bài tập: RNN & LSTM for sentiment analysis

So sánh **SimpleRNN, LSTM, Linear SVM và Logistic Regression** trên IMDB.
Notebook và báo cáo bằng tiếng Việt. Notebook tự chạy độc lập, không cần các
file Python trong `scripts/` khi tải lên Colab.

## Chạy bài đầy đủ trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook**,
   tải [IMDB_Sentiment_Analysis.ipynb](IMDB_Sentiment_Analysis.ipynb) lên.
2. Chọn **Runtime → Change runtime type → GPU**. Giữ `RUN_MODE = "full"`.
3. Chọn **Runtime → Run all**. Kiểm tra dòng GPU trong output phần môi trường.
   Nếu không có GPU, notebook vẫn chạy CPU nhưng sẽ chậm hơn.
4. Khi hoàn tất, tải `results/IMDB_full_results.zip` bằng bảng Files của Colab
   (menu ba chấm → Download). Hoặc đặt `DOWNLOAD_RESULTS=True` trước khi Run all.
5. Chọn **File → Download → Download .ipynb** để lưu notebook có output.
   Báo cáo kết quả nằm trong ZIP. Dữ liệu/mô hình trong `/content` có thể mất
   khi runtime bị ngắt; hãy tải kết quả trước khi đóng phiên.

Colab thường đã có các thư viện. Nếu báo thiếu package, xem ô hướng dẫn thư viện
trong notebook. Không cần API key hay quyền truy cập Google Drive.

## Nội dung bàn giao

- [Notebook chính](IMDB_Sentiment_Analysis.ipynb): dữ liệu, lý thuyết, huấn luyện,
  chọn cấu hình bằng validation, đánh giá test, biểu đồ, ví dụ lỗi và dự đoán thử.
- [Báo cáo đầy đủ](Bao_cao_IMDB.md): lý thuyết và phương pháp đã soạn sẵn;
  **kết quả đầy đủ đang chờ Run all trên Colab**, không có số liệu giả lập.
- [Notebook kiểm tra nhanh đã chạy](results/quick/IMDB_Sentiment_Analysis_quick.ipynb)
  và [báo cáo kiểm tra nhanh](results/quick/Bao_cao_IMDB_quick.md).
  Quick chỉ kiểm tra pipeline, không dùng thay kết quả full của bài nộp.

## Thiết kế và đầu ra

Giữ 25.000 mẫu test chính thức; tách train gốc thành 20.000 train / 5.000 validation,
seed 42, cân bằng nhãn. Chỉ học TF-IDF/vocabulary trên train. Baseline thử
`C=[0.1,1,10]` theo F1 validation. Mạng chọn checkpoint bằng validation loss,
tối đa 10 epoch, patience 2. Test không được dùng để chọn tham số.

Ưu tiên dữ liệu Stanford `.tar.gz`; khi máy chủ không truy cập được, notebook
dùng bản `stanfordnlp/imdb` trên Hugging Face với split train/test có sẵn.
Hai nguồn có cách đánh ID/thứ tự dòng khác nhau; nguồn và SHA-256 từng file được
ghi lại để tái lập. Các file `archive.zip` / `archive/IMDB Dataset.csv` sẵn có
không bị thay đổi; CSV thiếu split chính thức nên không được dùng để chia lại test.

Mỗi chế độ ghi vào thư mục riêng `results/full/` hoặc `results/quick/`:

| Tệp | Nội dung |
|---|---|
| `metrics.csv` | Accuracy, Precision, Recall, F1, ROC-AUC, thời gian và thiết bị |
| `baseline_validation.csv` | F1 validation và thời gian của từng C |
| `history_RNN.csv`, `history_LSTM.csv` | Loss/accuracy theo epoch |
| `data_summary.csv`, `split_manifest.csv` | Phân bố, độ dài và ID/nhãn từng split |
| `test_predictions.csv` | Dự đoán/điểm của bốn mô hình trên cùng test |
| `error_examples.csv`, `errors_by_length.csv` | Ví dụ lỗi và thống kê theo độ dài |
| `config.json`, `environment.json`, `checks.json` | Cấu hình, phiên bản, phần cứng, nguồn/checksum và kiểm tra |
| `figures/` | Bốn hình PNG: dữ liệu, learning curves, so sánh, confusion matrices |
| `models/` | TF-IDF, baseline joblib, vocabulary JSON và hai checkpoint Keras |
| `Bao_cao_IMDB.md` hoặc `Bao_cao_IMDB_quick.md` | Báo cáo tự điền từ kết quả thực |

Full còn cập nhật `Bao_cao_IMDB.md` tại thư mục làm việc. Quick không ghi đè
báo cáo này. Thời gian fit được tách khỏi tiền xử lý; baseline tính tổng ba C.
Baseline chạy CPU, mạng có thể chạy GPU, vì vậy cần đọc thời gian cùng thông tin
phần cứng. Baseline dùng toàn bộ văn bản, mạng giữ 300 token đầu; đây là giới hạn
cần nêu khi so sánh. SVM xuất margin, không coi margin là xác suất.

Mô hình `.keras` nhận chuỗi số nguyên. Để dùng lại, làm sạch như hàm `clean_text`
trong notebook, phục hồi `TextVectorization` từ `vocabulary.json` với
`standardize=None`, `output_sequence_length=300`, rồi nạp checkpoint. Với baseline,
làm sạch cùng cách, nạp TF-IDF rồi classifier. Hàm `predict_reviews` trong notebook
minh họa dự đoán trực tiếp khi các mô hình đã được huấn luyện.

## Kiểm tra nhanh tại máy cá nhân

Cài thư viện khi cần:

```powershell
python -m pip install -r requirements.txt
python scripts/run_quick_check.py
```

Script thực thi chính notebook, đổi sang quick trong bộ nhớ, dùng dữ liệu IMDB
thật (1.000 train / 200 validation / 1.000 test, 1 epoch). Script kiểm tra thêm
nhánh đọc archive Stanford bằng fixture, xác thực đầu ra và bảo đảm không ghi đè
báo cáo full. Notebook chính vẫn giữ chế độ full, không chứa kết quả quick.
Các cache Jupyter dùng riêng trong `.runtime/`.

Nếu cần chỉnh nguồn sinh notebook, sửa `scripts/build_notebook.py`, sau đó chạy:

```powershell
python scripts/build_notebook.py
```

Lệnh này tạo lại notebook; không ghi đè báo cáo đã tồn tại. Không có kiểm tra
accuracy tối thiểu vì kết quả phụ thuộc huấn luyện thực tế; các kiểm tra tập trung
vào tính đúng đắn của dữ liệu, phân chia, metric và khả năng chạy lại pipeline.

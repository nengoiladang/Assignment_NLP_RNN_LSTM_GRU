# Phân tích cảm xúc IMDB

So sánh SimpleRNN, LSTM, SVM và Logistic Regression trong notebook
`IMDB_Sentiment_Analysis.ipynb`.

## Cách chạy

1. Mở Terminal/PowerShell tại thư mục gốc dự án và cài thư viện:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Mở thư mục dự án bằng VS Code, sau đó mở
   `IMDB_Sentiment_Analysis.ipynb`.
3. Chọn **Select Kernel → Python Environments**, chọn đúng môi trường Python
   đã cài thư viện ở bước 1.
4. Đặt `RUN_MODE` ở ô cấu hình:
   - `"full"`: huấn luyện và đánh giá đầy đủ, dùng cho bài nộp.
   - `"quick"`: kiểm tra nhanh với ít dữ liệu và 1 epoch; không dùng để kết luận.
5. Chọn **Run All** để chạy từ đầu đến cuối, sau đó **Ctrl+S** để lưu output.
   Khi đổi chế độ, cần chạy lại **Run All**.

Nếu dùng Jupyter thay cho VS Code, chạy các lệnh sau tại thư mục gốc rồi mở notebook:

```powershell
python -m pip install notebook
python -m notebook
```

## Dữ liệu và kết quả

- Dữ liệu được đọc từ `data/`; nếu chưa có, notebook tự tải và cần kết nối Internet.
- Bảng số liệu, biểu đồ và phần phân tích được hiển thị trong notebook.
- Kết quả và mô hình được lưu trong `results/full/` hoặc `results/quick/`.
- Có thể chạy bằng CPU; thời gian phụ thuộc phần cứng và số epoch dừng sớm.

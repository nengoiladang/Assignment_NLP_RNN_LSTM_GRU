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

**Trạng thái: chờ chạy FULL trên Google Colab GPU.**

Chưa có số liệu huấn luyện đầy đủ. Mở `IMDB_Sentiment_Analysis.ipynb`, giữ `RUN_MODE = "full"`, bật GPU và Run all. Ô xuất báo cáo tự cập nhật bảng Accuracy/Precision/Recall/F1/ROC-AUC, thời gian, biểu đồ và kết luận. Không sử dụng kết quả quick để thay thế kết quả full.

Kết quả kiểm tra nhanh, nếu đã chạy, nằm riêng trong [báo cáo quick](results/quick/Bao_cao_IMDB_quick.md).

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

# DILI Prediction Web App (Demo)

Ứng dụng web demo dự đoán nguy cơ tổn thương gan do thuốc (DILI) từ SMILES,
dựng lại theo mục 3.6 khóa luận tốt nghiệp Phan Chí Hải (mô hình MACCS 166 Keys + Random Forest).

Có 2 bản: **Flask** (`app.py`) và **Streamlit** (`streamlit_app.py`) — chọn 1 trong 2, dùng chung `model/dili_model.pkl`.

## Cách 1: Streamlit (khuyến nghị — dễ chạy & deploy public miễn phí)

### Chạy local
```bash
pip install -r requirements.txt
python train_model.py         # huấn luyện mô hình demo (chỉ cần chạy 1 lần)
streamlit run streamlit_app.py
```
Trình duyệt sẽ tự mở tại http://localhost:8501

### Deploy public miễn phí (không cần biết gì về server)
1. Đẩy toàn bộ thư mục này lên 1 repo GitHub (public hoặc private đều được).
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **New app**.
3. Chọn repo vừa tạo, nhánh `main`, file chính là `streamlit_app.py` → **Deploy**.
4. Sau ~1-2 phút sẽ có link public dạng `https://<tên-app>.streamlit.app` để chia sẻ.

> Lưu ý: `model/dili_model.pkl` cần được commit lên GitHub cùng repo (đã có sẵn trong file zip này) để app deploy dùng được ngay, không cần chạy lại `train_model.py` trên server.

## Cách 2: Flask

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
Mở trình duyệt tại http://127.0.0.1:5000

Flask **không deploy được trên Streamlit Cloud/GitHub Pages** — muốn có link public cho bản Flask cần dịch vụ như Render.com, Railway, hoặc PythonAnywhere.

## Lưu ý quan trọng

- Mô hình trong bản demo này được huấn luyện trên **~40 chất mẫu minh họa**, KHÔNG PHẢI
  bộ dữ liệu thật (~5.172 mẫu từ DILIrank/LiverTox/DILI-DB) trong khóa luận.
- Để dùng model thật: thay `demo_data` trong `train_model.py` bằng dữ liệu thật đã tiền xử lý
  (xem notebook `DILI_MXAINet_Reconstructed.ipynb` mục 1-2), chạy lại `train_model.py`, model
  mới sẽ tự động được `app.py` sử dụng.
- Có thể đổi feature/model sang RDKit_2D + XGBoost (mô hình cho AUC cao nhất trong khóa luận,
  0.8451) bằng cách sửa `train_model.py` và `app.py` tương ứng — hỏi mình nếu cần hỗ trợ.
- Đây là server phát triển (Flask debug), không dùng để triển khai thực tế/production.

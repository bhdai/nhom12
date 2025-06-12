# Ứng dụng dự đoán giá đấu giá máy xúc

## Tính năng

- Tải lên và xem trước dữ liệu CSV
- Thông tin chi tiết về dữ liệu (thống kê, giá trị thiếu, kiểu dữ liệu)
- Các biểu đồ tương tác (phân bố năm sản xuất, doanh số theo thời gian)
- Dự đoán giá đấu giá của các máy
- Phân tích tầm quan trọng của các đặc trưng
- Biểu đồ giải thích chi tiết cho từng dự đoán
- Xuất kết quả sang định dạng CSV

## Cài đặt

### Sử dụng requirements.txt

Cài đặt các gói cần thiết:

```
pip install -r requirements.txt
```

## Chạy ứng dụng

```
cd ui
uvicorn main:app --reload
```

Ứng dụng sẽ được chạy tại http://127.0.0.1:8000

## Cách sử dụng

1. Tải lên tệp CSV chứa dữ liệu máy xúc
2. Xem thông tin và biểu đồ phân tích dữ liệu
3. Nhấn "Tiến hành dự đoán" để tạo dự đoán giá
4. Khám phá kết quả:
   - Xem biểu đồ tầm quan trọng của các đặc trưng
   - Nhấn "Explain" trên từng dự đoán để xem các yếu tố ảnh hưởng đến dự đoán đó
   - Xuất kết quả sang CSV nếu cần

## Hình ảnh minh họa

![Image](https://github.com/user-attachments/assets/6f3aa743-4586-4328-96c6-b28de252b935)

![Image](https://github.com/user-attachments/assets/7b0fb0be-970f-4964-9262-654ec44901ab)

![Image](https://github.com/user-attachments/assets/abeb7db8-efa8-45e2-9e93-62e3547605ab)

![Image](https://github.com/user-attachments/assets/063b248f-5a33-454e-ba98-60e0837cb426)

![Image](https://github.com/user-attachments/assets/236de79c-8db2-46db-97ed-068edf8cefe9)

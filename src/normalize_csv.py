import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv("project/models/optimized_test_predictions.csv")

# Xóa cột cuối cùng (theo index hoặc tên)
df = df.iloc[:, :-1]  # Xóa cột cuối cùng bất kể tên là gì

# Nếu muốn chắc chắn xóa theo tên:
# df.drop(columns=["interval_width"], inplace=True)

# Lưu lại file mới
df.to_csv("submission.csv", index=False)

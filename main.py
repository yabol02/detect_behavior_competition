import time
from pathlib import Path
from preprocess import load_ts_data, preprocess_pipeline

data_dir = Path("./data")
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

start_time = time.time()
train_lf = load_ts_data(train_path, null_values="-1.0")
print(f"Train data loading time: {time.time() - start_time:.2f} seconds")
start_time = time.time()
test_lf = load_ts_data(test_path, null_values="-1.0")
print(f"Test data loading time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
train_df = train_lf.pipe(preprocess_pipeline)
print(train_df.head(5))
print(f"Preprocessing train time: {time.time() - start_time:.2f} seconds")
start_time = time.time()
test_df = test_lf.pipe(preprocess_pipeline)
print(test_df.head(5))
print(f"Preprocessing test time: {time.time() - start_time:.2f} seconds")

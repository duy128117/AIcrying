import numpy as np
import os   
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import argparse

def load_features(path):
    """Tải dữ liệu từ file .npz"""
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']
    classes = data['classes']
    return X, y, classes

def main(args):
    # 1. Tải dữ liệu
    X, y, classes = load_features(args.features)
    num_classes = len(classes)
    print(f"Loaded {X.shape[0]} samples. Classes: {classes}")

    # 2. TÁI TẠO LẠI BỘ TEST
    # Logic này phải giống hệt file train.py để đảm bảo
    # chúng ta đang đánh giá trên đúng bộ Test đã tách ra.
    
    # Xáo trộn (phải giống train.py)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]; y = y[idx]
    y = y.astype('int32')

    # Tách test set (phải giống train.py)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, # 0.1
        random_state=42, 
        stratify=y
    )
    
    # Chuyển y_test sang one-hot
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"Recreated test set: {X_test.shape[0]} samples")

    # 3. Tải mô hình đã huấn luyện
    print(f"Loading model from {args.model}...")
    try:
        model = keras.models.load_model(args.model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model file exists and you have trained the model.")
        return

    # 4. Đánh giá mô hình
    print("Evaluating model on test set...")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=1)
    
    print("\n" + "="*20)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("="*20)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, 
                        default="../dataset/features.npz", 
                        help="Đường dẫn đến file features.npz")
    parser.add_argument("--model", type=str, 
                        default="../model/best_model.keras", 
                        help="Đường dẫn đến file model .keras đã lưu (vd: best_model.keras)")
    parser.add_argument("--test_size", type=float, 
                        default=0.1, 
                        help="Tỷ lệ test set (phải GIỐNG HỆT file train.py)")
    args = parser.parse_args()
    main(args)
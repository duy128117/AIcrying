import numpy as np
import os   
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def build_model(input_shape, num_classes, weights="imagenet"):
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        alpha=1.0
    )
    base.trainable = True
    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=base.input, outputs=outputs)
    return model

def load_features(path):
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']
    classes = data['classes']
    return X, y, classes

def plot_training_history(history, save_path):
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ lịch sử huấn luyện vào: {save_path}")

    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")
        print("Lưu ý: Nếu huấn luyện dừng lại chỉ sau 1 epoch, có thể không đủ dữ liệu để vẽ biểu đồ.")

def main(args):
    X, y, classes = load_features(args.features)
    num_classes = len(classes)
    print(f"Đã tải {X.shape[0]} mẫu, với {num_classes} lớp: {classes}")
    print(f"Hình dạng dữ liệu đầu vào (Input shape): {X.shape[1:]}")

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    y = y.astype('int32')

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size/(1-args.test_size), 
        random_state=42, stratify=y_temp
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    input_shape = X.shape[1:] 
    model = build_model(input_shape, num_classes, weights="imagenet")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(args.checkpoint_dir)
    
    model_path = os.path.join(args.checkpoint_dir, "best_model.keras")
    
    ckpt = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    
    es = EarlyStopping(
        monitor='val_accuracy',
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    rl = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    )

    print("\nBắt đầu huấn luyện...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, es, rl]
    )

    print("\nĐang đánh giá mô hình tốt nhất trên bộ dữ liệu test...")
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    model.save(os.path.join(args.checkpoint_dir, "final_model.keras"))
    np.save(os.path.join(args.checkpoint_dir, "classes.npy"), classes)
    
    plot_history_path = os.path.join(args.checkpoint_dir, "training_history.png")
    plot_training_history(history, plot_history_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="../dataset/features.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="../model")
    parser.add_argument("--epochs", type=int, default=80) 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.1) 
    parser.add_argument("--val_size", type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)
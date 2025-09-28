# convert_to_tflite.py (đã sửa đổi)
import tensorflow as tf
import keras  # Thêm thư viện keras
import numpy as np
import argparse
import os

def representative_dataset_gen(npy_path, num_samples=100):
    data = np.load(npy_path, allow_pickle=True)['X']
    def gen():
        for i in range(min(num_samples, len(data))):
            arr = data[i:i+1].astype(np.float32)
            yield [arr]
    return gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model", type=str, required=True)
    parser.add_argument("--out", type=str, default="model.tflite")
    parser.add_argument("--features", type=str, default="../dataset/features.npz", help="for representative dataset")
    parser.add_argument("--quantize", action='store_true', help="apply int8 quantization")
    args = parser.parse_args()

    # SỬA Ở ĐÂY: Dùng keras.models.load_model thay vì tf.keras
    model = keras.models.load_model(args.keras_model)
    
    # Phần còn lại giữ nguyên vì TFLiteConverter là công cụ của TensorFlow
    # và nó hoàn toàn tương thích với mô hình Keras.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        rep = representative_dataset_gen(args.features, num_samples=200)
        converter.representative_dataset = rep
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite:", args.out)
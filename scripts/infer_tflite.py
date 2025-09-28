# infer_tflite.py
import numpy as np, argparse, librosa
import tensorflow as tf
from preprocess_audio import preprocess_audio

def reshape_to_image(feature_vector, target_shape=(32, 32, 3)):
    flat_length = np.prod(target_shape[:-1])
    if len(feature_vector) < flat_length:
        feature_vector = np.pad(feature_vector, (0, flat_length - len(feature_vector)), mode='constant')
    elif len(feature_vector) > flat_length:
        feature_vector = feature_vector[:flat_length]
    img = feature_vector.reshape(target_shape[:-1])
    img = np.stack([img] * target_shape[-1], axis=-1)
    return (img - np.mean(img)) / (np.std(img) + 1e-9)

def extract_features(y, sr):
    feature = preprocess_audio(y, fs=sr)
    return reshape_to_image(feature)

def load_tflite_model(path):
    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    inp_details = inter.get_input_details()
    out_details = inter.get_output_details()
    return inter, inp_details, out_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True, help="Path to TFLite model")
    parser.add_argument("--wav", required=True, help="Path to wav file")
    parser.add_argument("--classes", type=str, default="../model/classes.npy", help="Path to classes.npy")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=2.0)
    args = parser.parse_args()

    inter, inp_det, out_det = load_tflite_model(args.tflite)
    classes = np.load(args.classes, allow_pickle=True)

    y, sr = librosa.load(args.wav, sr=args.sr, mono=True)
    seg_len = int(args.segment_sec * args.sr)
    idx = 0

    predictions = []
    all_probs = []

    while idx + seg_len <= len(y):
        seg = y[idx: idx+seg_len]
        img = extract_features(seg, sr)
        x = np.expand_dims(img, axis=0).astype(np.float32)

        if inp_det[0]['dtype'] == np.int8:
            in_scale, in_zero_point = inp_det[0]['quantization']
            x = (x / in_scale + in_zero_point).astype(np.int8)

        inter.set_tensor(inp_det[0]['index'], x)
        inter.invoke()
        out = inter.get_tensor(out_det[0]['index'])

        if out_det[0]['dtype'] == np.int8:
            o_scale, o_zero = out_det[0]['quantization']
            out = (out.astype(np.float32) - o_zero) * o_scale

        prob = tf.nn.softmax(out[0]).numpy()
        cls = np.argmax(prob)

        print(f"Segment {idx}/{len(y)} -> class {classes[cls]} ({cls}), probs {prob}")

        predictions.append(cls)
        all_probs.append(prob)

        idx += int(seg_len * 0.5)

    predictions = np.array(predictions)
    all_probs = np.array(all_probs)

    final_cls_majority = np.bincount(predictions).argmax()
    mean_probs = all_probs.mean(axis=0)
    final_cls_mean = np.argmax(mean_probs)

    print("\n===== FINAL RESULT =====")
    print(f"Majority vote -> {classes[final_cls_majority]} ({final_cls_majority})")
    print(f"Mean probs    -> {classes[final_cls_mean]} ({final_cls_mean}), probs={mean_probs}")
# Cry Detection with CNN on Mel-Spectrograms

An end-to-end deep learning project for classifying audio into **cry** and **not_crying** using Mel-spectrogram images and a **MobileNetV2**-based neural network.

The project covers the full pipeline:

- audio dataset preparation
- spectrogram feature extraction
- model training and evaluation
- TensorFlow Lite conversion
- benchmark and real-time inference from microphone or WAV files

---

## Project Goal

This project was built to detect crying sounds from audio recordings in a practical way. Instead of training directly on raw waveforms, the audio is converted into 2D Mel-spectrogram images and classified with a lightweight image-based CNN architecture.

This makes the solution suitable for:

- edge devices
- mobile deployment
- embedded AI experiments
- real-time audio monitoring

---

## Key Features

- **Audio preprocessing** with `librosa`
- **Mel-spectrogram generation** from 2-second audio segments
- **Noise augmentation** for the `cry` class using background audio
- **Transfer learning** with `MobileNetV2`
- **Train/validation/test split** with fixed random seed for reproducibility
- **Keras model export** (`.keras`)
- **TFLite conversion** with optional **INT8 quantization**
- **Benchmarking** of model size, speed, and accuracy
- **Live inference** from microphone or WAV input

---

## Repository Structure

```text
ai-git/
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── convert_to_tflite.py
│   ├── benchmark.py
│   ├── live_from_wav.py
│   └── infer_tflite.py
└── test_audio/
```

> Note: Depending on your local setup, you may also have `dataset/` and `model/` folders outside of the GitHub repo if they are too large to upload.

---

## Dataset Overview

The dataset is organized into class folders under `dataset/raw/`.

Typical structure:

```text
dataset/raw/
├── cry/
└── not_crying/
    ├── airplane/
    ├── breathing/
    ├── brushing_teeth/
    ├── can_opening/
    ├── car_horn/
    ├── cat/
    ├── chirping_birds/
    ├── church_bells/
    ├── clapping/
    ├── clock_alarm/
    ├── clock_tick/
    ├── common voice/
    ├── coughing/
    ├── dog/
    ├── fan/
    ├── footsteps/
    ├── helicopter/
    ├── insects/
    ├── keyboard_typing/
    ├── laughing/
    ├── mouse_click/
    ├── rain/
    ├── rooster/
    ├── sneezing/
    ├── snoring/
    ├── thunderstorm/
    ├── train/
    ├── vacuum_cleaner/
    ├── voice-newlive/
    ├── washing_machine/
    └── whisper/
```

The preprocessing script also supports a `background_noises/` folder to mix noise into cry samples during feature generation.

---

## Model Pipeline

1. Load `.wav` files
2. Split audio into fixed-length segments
3. Convert each segment to a Mel-spectrogram image
4. Resize to `128 x 128`
5. Duplicate channel to create `128 x 128 x 3`
6. Train `MobileNetV2` classifier
7. Save Keras model and class labels
8. Convert to TFLite for deployment

---

## Requirements

Main dependencies:

- Python 3.10+
- TensorFlow 2.20+
- Keras 3+
- librosa
- numpy
- scikit-learn
- matplotlib
- soundfile
- sounddevice
- Pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1) Prepare features

Convert raw audio into a compressed feature file:

```bash
python scripts/prepare_dataset.py --dataset_dir ../dataset/raw --out ../dataset/features.npz
```

What this script does:

- loads `.wav` files
- segments audio into 2-second windows
- optionally adds background noise to `cry` samples
- converts each segment to a normalized spectrogram image
- saves `X`, `y`, and `classes` into `features.npz`

---

### 2) Train the model

```bash
python scripts/train.py --features ../dataset/features.npz --checkpoint_dir ../model
```

Outputs:

- `best_model.keras`
- `final_model.keras`
- `classes.npy`
- `training_history.png`

Training uses:

- `MobileNetV2` backbone
- Adam optimizer
- early stopping
- learning rate reduction on plateau
- validation split and test split with fixed seed

---

### 3) Evaluate the trained model

```bash
python scripts/evaluate.py --features ../dataset/features.npz --model ../model/best_model.keras
```

This script recreates the same test split used during training and reports test loss and accuracy.

---

### 4) Convert to TFLite

```bash
python scripts/convert_to_tflite.py --keras_model ../model/best_model.keras --out ../model/model.tflite
```

For INT8 quantization:

```bash
python scripts/convert_to_tflite.py --keras_model ../model/best_model.keras --out ../model/model_quant_int8.tflite --features ../dataset/features.npz --quantize
```

---

### 5) Benchmark Keras vs TFLite

```bash
python scripts/benchmark.py --keras_model ../model/best_model.keras --tflite ../model/model_quant_int8.tflite --features ../dataset/features.npz
```

This compares:

- model file size
- inference speed
- test accuracy

---

### 6) Run inference on a WAV file

```bash
python scripts/infer_tflite.py --tflite ../model/model_quant_int8.tflite --wav test_audio/sample.wav --classes ../model/classes.npy
```

This script:

- splits the WAV file into segments
- predicts each segment
- aggregates predictions by majority vote and mean probabilities

---

### 7) Live microphone inference

```bash
python scripts/live_from_wav.py --tflite ../model/model_quant_int8.tflite --classes ../model/classes.npy
```

The script records audio from the microphone, converts it to spectrograms, and prints the final predicted class.

---

## Technical Details

- **Input sample rate:** 16 kHz
- **Segment length:** 2 seconds
- **Spectrogram size:** `128 x 128`
- **Channels:** 3-channel image input
- **Base model:** `MobileNetV2`
- **Loss function:** categorical cross-entropy
- **Metrics:** accuracy

---

## Why This Approach

Using spectrogram images allows the model to learn time-frequency patterns in audio, which works well for distinguishing crying from many background sounds.

Benefits:

- simpler training than raw waveform models
- easy to visualize audio patterns
- compatible with image CNN architectures
- good path to mobile/edge deployment via TFLite

---

## My Contribution / Project Highlights

If you want to present this on GitHub for recruiters, these are the strongest points to mention:

- designed a full audio AI pipeline from preprocessing to deployment
- handled data augmentation with real background noise
- built a reproducible training and evaluation workflow
- exported the model to TFLite for lightweight inference
- added scripts for offline, benchmark, and live microphone use cases

---

## Results

> Replace this section with your actual experiment results before publishing.

Suggested format:

| Metric | Keras | TFLite INT8 |
|---|---:|---:|
| Test Accuracy | xx.xx% | xx.xx% |
| Model Size | xx MB | xx MB |
| Inference Time | xx ms | xx ms |

---

## Future Improvements

- add a web or mobile demo UI
- experiment with other audio backbones
- improve class imbalance handling
- add confusion matrix and classification report
- package the pipeline into a single CLI
- deploy on Raspberry Pi or Android

---

## License

Add a license if you plan to publish this publicly on GitHub.

---

## Contact

If this is for your portfolio, add:

- your name
- email
- LinkedIn
- GitHub profile

---

## Short Vietnamese Summary

Dự án này xây dựng hệ thống nhận diện tiếng khóc từ âm thanh bằng cách chuyển audio sang Mel-spectrogram và phân loại bằng MobileNetV2. Ngoài mô hình Keras, dự án còn hỗ trợ xuất sang TFLite, benchmark, và suy luận realtime từ micro hoặc file WAV.
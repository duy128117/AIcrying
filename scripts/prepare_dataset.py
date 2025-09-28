# prepare_dataset.py (phiên bản không dùng cv2)
import os, glob
import numpy as np
import librosa
import argparse
from tqdm import tqdm
import random
# MỚI: Dùng thư viện Pillow thay cho cv2
from PIL import Image

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def segment_audio(y, sr, seg_seconds=2.0, hop_seconds=None):
    seg_len = int(seg_seconds * sr)
    if hop_seconds is None: hop = seg_len
    else: hop = int(hop_seconds * sr)
    if len(y) < seg_len:
        y = np.pad(y, (0, seg_len - len(y)), mode='constant')
    starts = list(range(0, max(1, len(y) - seg_len + 1), hop))
    segments = [y[s:s+seg_len] for s in starts]
    return segments

# --- HÀM ĐƯỢC CẬP NHẬT: Không còn dùng cv2 ---
def audio_to_spectrogram_image(audio_segment, sr=16000, target_shape=(128, 128)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Chuẩn hóa về 0-255
    img_normalized = (log_mel_spectrogram - log_mel_spectrogram.min()) / (log_mel_spectrogram.max() - log_mel_spectrogram.min())
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    # Dùng Pillow để resize
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize(target_shape, Image.Resampling.LANCZOS)
    img_resized = np.array(pil_img)
    
    # Dùng Numpy để chuyển thành ảnh 3 kênh
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    
    # Chuẩn hóa ảnh cho mô hình
    img_rgb = img_rgb.astype(np.float32) / 255.0
    
    return img_rgb

def add_noise_manual(signal, noise_files, sr, min_snr_db=10.0, max_snr_db=40.0):
    try:
        noise_path = random.choice(noise_files)
        noise, _ = librosa.load(noise_path, sr=sr)
        if len(noise) < len(signal):
            noise = np.pad(noise, (0, len(signal) - len(noise)), mode='wrap')
        start_idx = random.randint(0, len(noise) - len(signal))
        noise_segment = noise[start_idx : start_idx + len(signal)]
        signal_rms = np.sqrt(np.mean(signal**2))
        noise_rms = np.sqrt(np.mean(noise_segment**2))
        if noise_rms == 0 or signal_rms == 0: return signal
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr = 10 ** (snr_db / 20.0)
        scale = signal_rms / (noise_rms * snr)
        noise_scaled = noise_segment * scale
        return signal + noise_scaled
    except Exception:
        return signal

def main(args):
    classes = sorted([d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))])
    
    noise_dir = os.path.join(os.path.dirname(args.dataset_dir), 'background_noises')
    noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    if noise_files: print(f"Tìm thấy {len(noise_files)} file nhiễu nền.")

    X_list = []
    y_list = []
    target_shape = (128, 128)
    
    for idx, cls in enumerate(classes):
        files = glob.glob(os.path.join(args.dataset_dir, cls, "**/*.wav"), recursive=True)
        print(f"Processing class {cls}: {len(files)} files")
        for f in tqdm(files):
            try:
                y, sr = load_audio(f, sr=args.sr)
            except Exception as e:
                print("skip", f, e); continue
            
            segments = segment_audio(y, sr, seg_seconds=args.segment_sec, hop_seconds=args.segment_sec * args.overlap)
            
            for seg in segments:
                if noise_files and cls == 'cry' and random.random() < 0.7:
                    seg = add_noise_manual(seg, noise_files, sr)
                
                img = audio_to_spectrogram_image(seg, sr, target_shape)
                
                X_list.append(img)
                y_list.append(idx)
                
    X = np.array(X_list)
    y = np.array(y_list)
    print("Saving features to", args.out)
    np.savez_compressed(args.out, X=X, y=y, classes=classes)
    print("Done. X shape:", X.shape, "y shape:", y.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/raw")
    parser.add_argument("--out", type=str, default="../dataset/features.npz")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--segment_sec", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
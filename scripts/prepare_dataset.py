import os, glob
import numpy as np
import librosa
import argparse
from tqdm import tqdm
import random
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

def audio_to_spectrogram_image(audio_segment, sr=16000, target_shape=(128, 128)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128) 
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    spec_min = log_mel_spectrogram.min()
    spec_max = log_mel_spectrogram.max()
    
    if spec_max == spec_min:
    
        img_normalized = np.zeros(log_mel_spectrogram.shape, dtype=np.float32)
    else:
        img_normalized = (log_mel_spectrogram - spec_min) / (spec_max - spec_min)
 
    
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize(target_shape, Image.Resampling.LANCZOS)
    img_resized = np.array(pil_img)
    
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    
    img_rgb = img_rgb.astype(np.float32) / 255.0 
    
    return img_rgb

def add_noise_manual(signal, noise_files, sr, min_snr_db=10.0, max_snr_db=40.0):
    try:
        noise_path = random.choice(noise_files)
        noise, _ = librosa.load(noise_path, sr=sr)
        
        if len(noise) < len(signal):
            repeats = int(np.ceil(len(signal) / len(noise)))
            noise = np.tile(noise, repeats)
            
        start_idx = random.randint(0, len(noise) - len(signal))
        noise_segment = noise[start_idx : start_idx + len(signal)]
        
        signal_rms = np.sqrt(np.mean(signal**2))
        noise_rms = np.sqrt(np.mean(noise_segment**2))
        
        if noise_rms == 0 or signal_rms == 0: 
            return signal
            
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr = 10 ** (snr_db / 20.0)
        scale = signal_rms / (noise_rms * snr)
        noise_scaled = noise_segment * scale
        return signal + noise_scaled
    except Exception as e:
        print(f"\nCảnh báo: Không thể thêm nhiễu. Lỗi: {e}")
        return signal

def main(args):
    classes = sorted([d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))])
    
    print(f"Đã phát hiện {len(classes)} lớp: {classes}")

    noise_dir = os.path.join(os.path.dirname(args.dataset_dir), 'background_noises')
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    
    if noise_files: 
        print(f"Tìm thấy {len(noise_files)} file nhiễu nền.")
    else:
        print("Không tìm thấy thư mục 'background_noises' hoặc không có file .wav nào bên trong.")

    X_list = []
    y_list = []
    target_shape = (128, 128)
    
    for idx, cls in enumerate(classes):
        class_path = os.path.join(args.dataset_dir, cls)
        if not os.path.isdir(class_path):
            continue
            
        files = glob.glob(os.path.join(class_path, "**/*.wav"), recursive=True)
        print(f"Processing class '{cls}' (ID: {idx}): {len(files)} files found.") 
        
        for f in tqdm(files, desc=f"Class {cls}"): 
            try:
                y, sr = load_audio(f, sr=args.sr)
                
                if len(y) < sr * 0.1:
                    continue
                    
            except Exception as e:
                print(f"\nLỗi khi tải file, bỏ qua: {f}. Lỗi: {e}")
                continue
            
            
            segments = segment_audio(y, sr, seg_seconds=args.segment_sec, hop_seconds=args.segment_sec * args.overlap)
            
            for seg in segments:
                if noise_files and cls == 'cry' and random.random() < 0.7:
                    seg = add_noise_manual(seg, noise_files, sr)
                
                img = audio_to_spectrogram_image(seg, sr, target_shape)
                
                X_list.append(img)
                y_list.append(idx)
                
    if not X_list:
        print("LỖI: Không tìm thấy hoặc không xử lý được tệp âm thanh nào. Tệp features.npz sẽ trống.")
        return

    X = np.array(X_list)
    y = np.array(y_list)
    
    ensure_dir(os.path.dirname(args.out))
    
    print(f"Saving {X.shape[0]} features to {args.out}...")
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
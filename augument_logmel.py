import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa
import matplotlib.pyplot as plt
import cv2

# Cấu hình
#test 2 loại min max
#test các loại mode  'bicubic' ' bilinear' 'area'
sample_rate = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def process_audio_file(input_path, output_path=None, target_sample_rate=sample_rate, enable_augments=True):
    try:
        # 1. Load file âm thanh
        waveform, original_sample_rate = torchaudio.load(input_path)
        waveform = waveform.to(device)

        # 3. Resample về tần số mong muốn
        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=target_sample_rate).to(device)
            waveform = resampler(waveform)
            
        # Trích xuất đặc trưng gốc
        results = {
            'original': extract_log_mel_features(waveform, target_sample_rate)
        }
        
        # Chỉ thực hiện augmentation nếu được bật
        if enable_augments:
            # 3. Augmentation
            augments = {
                'vol_small': extract_log_mel_features(change_volume(waveform.clone(), 0.7, 1.0), target_sample_rate),
                'vol_large': extract_log_mel_features(change_volume(waveform.clone(), 1.0, 1.3), target_sample_rate),
                'pitch_down': extract_log_mel_features(pitch_shift(waveform.clone(), target_sample_rate, -2, 0), target_sample_rate),
                'pitch_up': extract_log_mel_features(pitch_shift(waveform.clone(), target_sample_rate, 0, 2), target_sample_rate),
                'time_slow': extract_log_mel_features(time_stretch(waveform.clone(), 0.8, 1.0), target_sample_rate),
                'time_fast': extract_log_mel_features(time_stretch(waveform.clone(), 1.0, 1.2), target_sample_rate),
                'noise_small': extract_log_mel_features(add_noise(waveform.clone(), 0.001, 0.005), target_sample_rate),
                'noise_big': extract_log_mel_features(add_noise(waveform.clone(), 0.005, 0.01), target_sample_rate)
            }
            results.update(augments)

            # 3. Augmentation combine 
            combined = waveform.clone()
            combined = change_volume(combined, 0.7, 1.3)
            combined = pitch_shift(combined, target_sample_rate, -2, 2)
            combined = time_stretch(combined, 0.8, 1.2)
            combined = add_noise(combined, 0.001, 0.01)
            combined = extract_log_mel_features(combined, target_sample_rate)
            results['combined'] = combined
        
        return results

    except Exception as e:
        raise RuntimeError(f"Lỗi khi xử lý file: {e}") from e


# ------------------------- Augmentation -------------------------

def add_noise(waveform, min_noise=0.001, max_noise=0.01):
    noise_factor = torch.FloatTensor(1).uniform_(min_noise, max_noise).to(waveform.device).item()
    noise = torch.randn_like(waveform) * noise_factor
    return waveform + noise

def pitch_shift(waveform, sample_rate, min_steps=-2, max_steps=2):
    n_steps = np.random.uniform(min_steps, max_steps)
    shifted = librosa.effects.pitch_shift(
        waveform.cpu().numpy().squeeze(), sr=sample_rate,
        n_steps=n_steps).astype(np.float32)
    return torch.from_numpy(shifted).to(waveform.device).unsqueeze(0)

def time_stretch(waveform, min_rate=0.8, max_rate=1.2):
    waveform_np = waveform.squeeze(0).cpu().numpy()
    rate = np.random.uniform(min_rate, max_rate)
    stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
    return torch.tensor(stretched, device=waveform.device).unsqueeze(0)

def change_volume(waveform, min_factor=0.7, max_factor=1.3):
    volume_factor = torch.FloatTensor(1).uniform_(min_factor, max_factor).to(waveform.device).item()
    return waveform * volume_factor


# ------------------------- Feature Extraction -------------------------

def normalize(spec):
    return (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
def _normalize(x):
        return (x - x.mean()) / (x.std() + 1e-9)

def extract_log_mel_features(waveform, sample_rate, n_mels=224, hop_length=1024, win_length=None, n_fft=2048):
    device = waveform.device
    
    # Tạo Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=lambda win_length: torch.hann_window(win_length, device=device),
        n_mels=n_mels,
        power=2.0
    ).to(device)

    mel_spec = mel_transform(waveform)  # (1, n_mels, time)
    log_mel = torch.log(mel_spec + torch.finfo(mel_spec.dtype).eps)

    
    # Tính delta features
    delta = torchaudio.functional.compute_deltas(log_mel)
    delta2 = torchaudio.functional.compute_deltas(delta)

    # Chuẩn hóa từng đặc trưng
    
    log_mel = _normalize(log_mel)
    delta = _normalize(delta)
    delta2 = _normalize(delta2)

    # Kết hợp và resize đúng cách
    combined = torch.cat([log_mel, delta, delta2], dim=0)  # (3, n_mels, time)
    
    # Sử dụng interpolate của PyTorch thay vì cv2.resize
    # Chú ý: mode 'bilinear' cho dữ liệu 2D (time-frequency)
    resized = F.interpolate(combined.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False) #(1,3,224,224)
    return resized.squeeze(0)  # (3, 224, 224)




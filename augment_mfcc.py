import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa
import matplotlib.pyplot as plt
import cv2
from ID_label import train_label, test_label, val_label

# Cấu hình
sample_rate = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_audio_file(input_path, output_path=None, target_sample_rate=sample_rate, enable_augments=True):
    try:
        waveform, original_sample_rate = torchaudio.load(input_path)
        waveform = waveform.to(device)

        if original_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=target_sample_rate).to(device)
            waveform = resampler(waveform)

        results = {
            'mfcc_original': extract_mfcc_segments_mean(waveform, target_sample_rate)
        }

        if enable_augments:
            augment_funcs = {
                'vol_small': lambda x: change_volume(x, 0.7, 1.0),
                'vol_large': lambda x: change_volume(x, 1.0, 1.3),
                'pitch_down': lambda x: pitch_shift(x, target_sample_rate, -2, 0),
                'pitch_up': lambda x: pitch_shift(x, target_sample_rate, 0, 2),
                'time_slow': lambda x: time_stretch(x, 0.8, 1.0),
                'time_fast': lambda x: time_stretch(x, 1.0, 1.2),
                'noise_small': lambda x: add_noise(x, 0.001, 0.005),
                'noise_big': lambda x: add_noise(x, 0.005, 0.01),
            }

            for name, func in augment_funcs.items():
                aug_waveform = func(waveform.clone())
                results[f'mfcc_{name}'] = extract_mfcc_segments_mean(aug_waveform, target_sample_rate)

            # Combined augmentation
            combined = waveform.clone()
            combined = change_volume(combined, 0.7, 1.3)
            combined = pitch_shift(combined, target_sample_rate, -2, 2)
            combined = time_stretch(combined, 0.8, 1.2)
            combined = add_noise(combined, 0.001, 0.01)
            results['mfcc_combined'] = extract_mfcc_segments_mean(combined, target_sample_rate)

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


def extract_mfcc(waveform, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': 128,
            'center': True,
            'power': 2.0,
        }
    ).to(waveform.device)

    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time)
    mfcc = mfcc.squeeze(0)           # (n_mfcc, time)
    mfcc_mean = mfcc.mean(dim=1, keepdim=True)  # (n_mfcc, 1)
    return mfcc_mean.cpu()


def extract_mfcc_segments_mean(waveform, sample_rate, segment_duration_sec=2.0, n_mfcc=13):
    segment_samples = int(segment_duration_sec * sample_rate)
    total_samples = waveform.shape[1]
    mfcc_features = []

    n_segments = total_samples // segment_samples
    if n_segments == 0:
        waveform = F.pad(waveform, (0, segment_samples - total_samples))
        n_segments = 1

    for i in range(n_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = waveform[:, start:end]

        mfcc_mean = extract_mfcc(segment, sample_rate, n_mfcc=n_mfcc)  # (n_mfcc, 1)
        mfcc_features.append(mfcc_mean)

    mfcc_stack = torch.stack(mfcc_features)  # (n_segments, n_mfcc, 1)

    mfcc_mean_over_segments = mfcc_stack.mean(dim=0)  # (n_mfcc, 1)

    return mfcc_mean_over_segments  # (n_mfcc, 1)
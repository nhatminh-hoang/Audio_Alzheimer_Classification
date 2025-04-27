import os
import torch
from tqdm import tqdm
from ID_label import train_ID, train_label, test_ID, test_label
from augument_logmel import process_audio_file

# Đường dẫn đến các file audio
train_CD = r"D:\code\python\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cd"
train_CC = r"D:\code\python\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cc"
test_audio_path = r"D:\code\python\ADReSS-IS2020-data\test\Full_wave_enhanced_audio"

# Tạo thư mục lưu output
output_dir = "./processed"
train_path = os.path.join(output_dir, "train")
test_path = os.path.join(output_dir, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Tách train_ID thành 2 phần: CD (từ đầu đến S156), CC (sau S156)
idx_split = train_ID.index("S156 ")

cd_IDs = train_ID[:idx_split + 1]
cd_labels = train_label[:idx_split + 1]

cc_IDs = train_ID[idx_split + 1:]
cc_labels = train_label[idx_split + 1:]

# Xử lý CD với tqdm
print("Processing CD files...")
for id, label in tqdm(zip(cd_IDs, cd_labels), total=len(cd_IDs)):
    audio_file = os.path.join(train_CD, f"{id.strip()}.wav")
    output_file = os.path.join(train_path, f"{id}.pt")

    tensor = process_audio_file(audio_file)
    torch.save({
        "features": tensor,
        "label": label,
        "source": "cd"
    }, output_file)

# Xử lý CC với tqdm
print("Processing CC files...")
for id, label in tqdm(zip(cc_IDs, cc_labels), total=len(cc_IDs)):
    audio_file = os.path.join(train_CC, f"{id.strip()}.wav")
    output_file = os.path.join(train_path, f"{id}.pt")

    tensor = process_audio_file(audio_file)
    torch.save({
        "features": tensor,
        "label": label,
        "source": "cc"
    }, output_file)

# Xử lý test với tqdm
print("Processing Test files...")
for id, label in tqdm(zip(test_ID, test_label), total=len(test_ID)):
    audio_file = os.path.join(test_audio_path, f"{id.strip()}.wav")
    output_file = os.path.join(test_path, f"{id}.pt")

    tensor = process_audio_file(audio_file)
    torch.save({
        "features": tensor,
        "label": label
    }, output_file)

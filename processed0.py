import os
import torch
from tqdm import tqdm
from ID_label import train_ID, train_label, test_ID, test_label,val_ID,val_label
from augument_logmel import process_audio_file

# Đường dẫn đến các file audio
train_CD = r"D:\code\python\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cd"
train_CC = r"D:\code\python\ADReSS-IS2020-data\train\Full_wave_enhanced_audio\cc"
test_audio_path = r"D:\code\python\ADReSS-IS2020-data\test\Full_wave_enhanced_audio"

# Tạo thư mục lưu output
output_dir = "./processed"
train_path = os.path.join(output_dir, "train")
val_path = os.path.join(output_dir, "val")
test_path = os.path.join(output_dir, "test")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

idx_split = 0  # Giá trị mặc định nếu không tìm thấy số 1
for i in range(len(train_label) - 1, -1, -1):  # Duyệt từ cuối về đầu
    if train_label[i] == 1:
        idx_split = i
        break


cd_IDs = train_ID[:idx_split + 1]
cd_labels = train_label[:idx_split + 1]

cc_IDs = train_ID[idx_split + 1:]
cc_labels = train_label[idx_split + 1:]

# Xử lý CD với tqdm
print("Processing CD files...")
for id, label in tqdm(zip(cd_IDs, cd_labels), total=len(cd_IDs)):
    audio_file = os.path.join(train_CD, f"{id.strip()}.wav")
    

    results = process_audio_file(audio_file)
    for key, tensor in results.items():
        output_file = os.path.join(train_path, f"{id}_{key}.pt")  # Thêm key vào tên file
        torch.save({
            "features": tensor,
            "label": label,
            "source": "cd"
        }, output_file)

# Xử lý CC với tqdm
print("Processing CC files...")
for id, label in tqdm(zip(cc_IDs, cc_labels), total=len(cc_IDs)):
    audio_file = os.path.join(train_CC, f"{id.strip()}.wav")
    

    results = process_audio_file(audio_file)
    for key, tensor in results.items():
        output_file = os.path.join(train_path, f"{id}_{key}.pt")  # Thêm key vào tên file
        torch.save({
            "features": tensor,
            "label": label,
            "source": "cc"
        }, output_file)

#-----------------------------------------------------------------------------------
idx_split = 0  # Giá trị mặc định nếu không tìm thấy số 1
for i in range(len(val_label) - 1, -1, -1):  # Duyệt từ cuối về đầu
    if val_label[i] == 1:
        idx_split = i
        break


cd_IDs = val_ID[:idx_split + 1]
cd_labels = val_label[:idx_split + 1]

cc_IDs = val_ID[idx_split + 1:]
cc_labels = val_label[idx_split + 1:]

print("Processing CD files...")
for id, label in tqdm(zip(cd_IDs, cd_labels), total=len(cd_IDs)):
    audio_file = os.path.join(train_CD, f"{id.strip()}.wav")
    



    results = process_audio_file(audio_file,enable_augments= False)
    for key, tensor in results.items():
        output_file = os.path.join(val_path, f"{id}_{key}.pt")  # Thêm key vào tên file
        torch.save({
            "features": tensor,
            "label": label,
            "source": "cd"
        }, output_file)

# Xử lý CC với tqdm
print("Processing CC files...")
for id, label in tqdm(zip(cc_IDs, cc_labels), total=len(cc_IDs)):
    audio_file = os.path.join(train_CC, f"{id.strip()}.wav")

    results = process_audio_file(audio_file,enable_augments= False)
    for key, tensor in results.items():
        output_file = os.path.join(val_path, f"{id}_{key}.pt")  # Thêm key vào tên file
        torch.save({
            "features": tensor,
            "label": label,
            "source": "cc"
        }, output_file)
#-----------------------------------------------------------------------------------

# Xử lý test với tqdm
print("Processing Test files...")
for id, label in tqdm(zip(test_ID, test_label), total=len(test_ID)):
    audio_file = os.path.join(test_audio_path, f"{id.strip()}.wav")

    
    results = process_audio_file(audio_file,enable_augments= False)
    for key, tensor in results.items():
        output_file = os.path.join(test_path, f"{id}_{key}.pt")  # Thêm key vào tên file
        torch.save({
            "features": tensor,
            "label": label,
        }, output_file)

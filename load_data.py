import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

class TorchDataDataset(Dataset):
    def __init__(self, folder_path, device='cpu', transform=None):
        """
        Args:
            folder_path (str): Đường dẫn đến thư mục chứa file .pt.
            device (str): Thiết bị lưu trữ ('cuda' hoặc 'cpu').
            transform (callable): Hàm augment dữ liệu (nếu cần).
        """
        self.device = device
        self.transform = transform
        self.file_paths = glob.glob(os.path.join(folder_path, "*.pt"))
        
        if not self.file_paths:
            raise ValueError(f"No .pt files found in {folder_path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Load dữ liệu
            data = torch.load(self.file_paths[idx], map_location=self.device,weights_only= True)
            
            # Xử lý features
            features = data['features']
            
            
            # Xử lý label
            label = torch.tensor(data['label'], dtype=torch.long)
            
            # Áp dụng transform (nếu có)
            if self.transform:
                features = self.transform(features)
                
            return features, label
        
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            # Trả về một mẫu ngẫu nhiên khác để tránh lỗi
            return self[random.randint(0, len(self) - 1)]
        

def get_dataloaders(train_folder, test_folder,val_folder, batch_size=4, num_workers=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo Dataset
    train_dataset = TorchDataDataset(train_folder, device=device)
    val_dataset = TorchDataDataset(val_folder,device=device)
    test_dataset = TorchDataDataset(test_folder, device=device)
    
    # Khởi tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        
        
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False # Không cần shuffle cho test
        
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False # Không cần shuffle cho test
        
    )
    
    return train_loader, val_loader,test_loader


train_folder = r"D:\code\python\processed\train"
test_folder = r"D:\code\python\processed\test"
val_folder = r"D:\code\python\processed\val"
    
    # Lấy DataLoader
train_loader,val_loader ,test_loader = get_dataloaders(train_folder ,val_folder , test_folder ,16)
    
    




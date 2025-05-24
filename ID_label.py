import os
import pandas as pd
import random



# tác dụng của 2 dòng này là để hiện thị toàn bộ dữ liệu trong dataframe nếu bạn k muốn có thể xóa 2 dòng dưới
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Định nghĩa đường dẫn
DATAPATH = "./data/ADReSS-IS2020-data"
TRAIN_PATH = DATAPATH + "/train"
TEST_PATH = DATAPATH + "/test"


FULLWAVE = "/Full_wave_enhanced_audio"
CHUNKSWAVE = "/Normalised_audio-chunks"
TRANSCRIPTION = "/transcription"
# Kiểm tra xem thư mục train có tồn tại không
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train directory not found at {TRAIN_PATH}")



# Đường dẫn đến metadata
AD_data_txt = "/cd_meta_data.txt"  # Dữ liệu Alzheimer (Label = 1)
NAD_data_txt = "/cc_meta_data.txt"  # Dữ liệu không bệnh (Label = 0)


train_AD_data = pd.read_csv(TRAIN_PATH + AD_data_txt, delimiter=';', skipinitialspace=True)  #đọc path , các cột phân cách bằng dấu ";" và Bỏ qua khoảng trắng thừa sau dấu phân cách
train_AD_data['Label '] = 1   #tạo cột Label và gán giá trị 1
train_AD_data = train_AD_data.drop(columns=['age', 'mmse', 'gender '], axis=1)  # xóa age , mmse , gender , axis = 1 nghĩa là xóa cột 

#tương tự cái trên nhưng với dữ liệu không bệnh
train_NAD_data = pd.read_csv(TRAIN_PATH + NAD_data_txt, delimiter=';', skipinitialspace=True)
train_NAD_data['Label '] = 0
train_NAD_data = train_NAD_data.drop(columns=['age', 'mmse', 'gender '], axis=1)


#kết hợp dữ liệu Alzheimer và không bệnh
train_df = pd.concat([train_AD_data, train_NAD_data], ignore_index=True)

#tương tự như train
test_df = pd.read_csv(DATAPATH + '/2020Labels.txt', delimiter=';', skipinitialspace=True) #label đã có sẵn trong file 2020Labels.txt
test_df = test_df.drop(columns=['age', 'mmse', 'gender'], axis=1)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


train_ID = train_df['ID'].tolist()
train_label = train_df['Label'].tolist()



ad_ids = [id for id, label in zip(train_ID, train_label) if label == 1]  
nad_ids = [id for id, label in zip(train_ID, train_label) if label == 0]

random.seed(42)  # Đặt seed để có thể tái tạo kết quả
val_ad_ids = random.sample(ad_ids, 15)
val_nad_ids = random.sample(nad_ids, 15)

val_ID = val_ad_ids + val_nad_ids
val_label = [1] * 15 + [0] * 15

train_ID_label = list(zip(train_ID, train_label))  # Gộp ID và label lại với nhau

# Giữ lại chỉ những phần tử mà ID không nằm trong val_ID
filtered_train = [(id, label) for id, label in train_ID_label if id not in val_ID]

# Tách lại thành train_ID và train_label mới
train_ID, train_label = zip(*filtered_train)
train_ID = list(train_ID)
train_label = list(train_label)

test_ID = test_df['ID'].tolist()
test_label = test_df['Label'].tolist()
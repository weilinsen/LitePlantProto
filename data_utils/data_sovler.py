import os
import csv

data_root = "/home/Jhin0324/Jhin/PlantDoc"
# data_root = "/home/Jhin0324/Jhin/local_maize"

label_names = os.listdir(data_root)
label_names = sorted(label_names)

# print(label_names)
# train_labels = label_names[:10]
# 本地玉米的设置
# train_labels = ['1', '2', '4', '5', '6', '8', '10', '11', '12','15']
# test_labels = ['0', '3', '7', '9', '13', '14']

# train_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#                 '13','14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
#                 '25', '26', '27']

# # 本地水稻的设置
# train_labels = ['0', '1', '2', '3', '4', '5', '7', '8', '11', '12',
#                 '13','14',  '16', '18', '19', '20',  '22', '23', '24',]
# test_labels = ['6', '10', '9', '15', '17', '21']



# #
# train_labels = ['0', '1', '2', '3', '5', '6', '7', '8', '9', '11', '12',
#                 '13', '15', '16', '17', '18', '20', '21', '23', '24', '25', '26']
# test_labels = ['28', '29', '30', '31', '32', '33', '34', '35', '36', '37']

#PlantDoc_split1
# train_labels = [ '1', '3','4', '5', '6', '7', '8', '9', '10', '12',
#                 '13', '14', '16', '17',  '19', '20', '21', '22', '23', '25', '26']
# test_labels = ['0', '2','11', '15', '18', '24']

# #PlantDoc_split2
# train_labels = ['0', '2', '3','4',  '6', '7', '9', '10', '11',
#                 '12', '15', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26']
# test_labels = [ '1', '5','8', '13', '14', '16']

# # 本地玉米
# train_labels = ['0', '2', '3','4',  '6', '7', '9', '10', '11',
#                 '12', '15']
# test_labels = [ '0', '2', '3','4', '6', '7', '8', '9', '10', '11', '12','14','15']

# 本地水稻
train_labels = ['0', '2', '3','4',  '6', '7', '9', '10', '11',
                '12', '15']
test_labels = [ '1', '3', '5', '8', '14', '15', '18', '20']

# test_labels = label_names[10:]
print(train_labels)
print(test_labels)

with open("CD_local_maize_train_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in train_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])

with open("CD_local_maize_test_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in test_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])
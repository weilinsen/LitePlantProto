# import os
# import csv
# import shutil
#
#
# def mkdir(path):
#     folder = os.path.exists(path)
#
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#         print
#         "---  new folder...  ---"
#         print
#         "---  OK  ---"
#
#     else:
#         print
#         "---  There is this folder!  ---"
#
#
# data_root = "/home/Jhin0324/Jhin/ip102_v1.1/test.txt"
# file = open(data_root)
# for line in file.readlines():
#     curLine = line.strip().split(" ")
#     img = curLine[0]
#     label = curLine[1]
#     print(img)
#     print(label)
#     mkdir("/home/Jhin0324/Jhin/IP102/"+label)
#     shutil.copyfile("/home/Jhin0324/Jhin/ip102_v1.1/images/"+img, "/home/Jhin0324/Jhin/IP102/"+label+"/"+img)

import os
import csv
import random
data_root = "/home/Jhin0324/Jhin/IP102"

label_names = os.listdir(data_root)

sample_num = 81
labels = random.sample(label_names, sample_num)
print(len(labels))
print(labels)
train_labels = labels[:60]
test_labels = labels[60:]
print(len(train_labels))
print(len(test_labels))

with open("IP102train_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in train_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])

with open("IP102test_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in test_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])
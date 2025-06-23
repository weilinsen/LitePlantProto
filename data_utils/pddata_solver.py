import os
import csv

data_root = "/home/Jhin0324/Jhin/PlantDoc"

label_names = os.listdir(data_root)
train_labels = label_names[:21]
test_labels = label_names[21:]
print(train_labels)
print(test_labels)

with open("pdtrain_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in train_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])

with open("pdtest_data.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])
    for label in test_labels:
        label_root = data_root + "/" + label
        pics = os.listdir((label_root))
        for pic in pics:
            writer.writerow([label_root+'/'+pic, label])
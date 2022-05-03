import os
import random
import shutil

source_path = '/home/rezasaed/scratch/database/'
source_file = '/home/rezasaed/files.txt'
dest_file = '/home/rezasaed/ManTraNet-pytorch/MantraNet/files.txt'

source_images_path = source_path + 'images/'
source_masks_path = source_path + 'masks/'
source_edges_path = source_path + 'edges/'

fake_images = []
real_images = []

with open(source_file, "r") as f:
    for line in f:
        imagePath = line.split()[0]
        imageFile = imagePath.split('/')[-1]

        if int(line.split()[-1]) == 0:
            real_images.append(imageFile)
        else:
            fake_images.append(imageFile)

print(len(fake_images))
print(len(real_images))

f = open(dest_file, "w")

for fake_image in fake_images:
    f.write(source_images_path + fake_image + ' ' + source_masks_path + fake_image + '\n')
for real_image in real_images:
    f.write(source_images_path + real_image + ' None' + '\n')

f.close()

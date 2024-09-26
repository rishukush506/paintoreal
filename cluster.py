import torch
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import cv2
from PIL import Image
import os
import extcolors
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.io import read_image 
import torch.nn as nn
from PIL import Image
import pickle 
from patchify import patchify
from sklearn.cluster import KMeans


segcolors = []
count = 0
for i in os.listdir('/home/ishika/kaushik/art2real/images/segmentation'):
      colors, pixel_count = extcolors.extract_from_path("/home/ishika/kaushik/art2real/images/segmentation/"+i)
      for j in range(len(colors)):
        segcolors.append(colors[j][0]) 

      print(count)  
      count+=1  


segkeys = list(set(segcolors))
imgdict = {key: [] for key in segkeys}


unfold = nn.Unfold(kernel_size=(16, 16), stride = 6)
fold = nn.Fold(output_size = (256, 256), kernel_size=(16, 16), stride = 6)

for i in os.listdir('/home/ishika/kaushik/art2real/images/segmentation'):
    img_seg = (Image.open('/home/ishika/kaushik/art2real/images/segmentation/'+i)) 
    img_segarr = np.asarray(img_seg)
    patches_seg = patchify(img_segarr,(16, 16, 3), 6)

    img = (Image.open('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)) 
    img_arr = np.asarray(img) 
    patches = patchify(img_arr,(16, 16, 3), 6)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            im = Image.fromarray(patches_seg[i][j][0].astype(np.uint8))
            im.save("img1.png")
            colors, pixel_count = extcolors.extract_from_path("img1.png")
            for j in range(len(colors)):
                if (colors[j][1]/ pixel_count) > 0.25:
                    if colors[j][0] not in imgdict:
                        imgdict[colors[j][0]] = [patches[i][j][0]]

                    else:
                        imgdict[colors[j][0]].append(patches[i][j][0])



with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(imgdict, f)
        
with open('saved_dictionary.pkl', 'rb') as f:
    img_dict = pickle.load(f)
    # print('patches', patches.shape)
    # imageseg = read_image('/home/ishika/kaushik/art2real/images/segmentation/'+i)
    # imageseg = imageseg.to(torch.float32)
    # image = read_image('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
    # image = image.to(torch.float32)

    # outputseg = unfold(imageseg)
    # outputimg = unfold(image)

    # # im = Image.fromarray((fold(outputseg).permute(1,2,0).cpu().numpy()).astype(np.uint8))
    # # im.save("img.png")

    # # im = Image.fromarray((imageseg.permute(1,2,0).cpu().numpy()).astype(np.uint8))
    # # im.save("img.png")
    # # break
    # listseg =  outputseg.permute(1,0).tolist()
    # listimg = outputimg.permute(1,0).tolist()
    # print(len(listseg[0]))
    # cnt = 0
    # for m in listseg:
    #     print(torch.Tensor(m).view(3, 16, 16).size())
    #     im = Image.fromarray((torch.Tensor(m).view(3, 16, 16).permute(1, 2 ,0).cpu().numpy()).astype(np.uint8))
    #     im.save("img.png")
    #     colors, pixel_count = extcolors.extract_from_path("img.png")
    #     for j in range(len(colors)):
    #           if (colors[j][1]/ pixel_count) > 0.25:
    #                imgdict[colors[j][0]].append(listimg[cnt])

    #     cnt+=1



final_dict = {key: [] for key in segkeys}
kmean = Kmeans(n_clusters = 2, random_state = 0, n_init = 'auto')
for keyvalues, imgvalues in zip(imgdict.keys(), imgdict.values()):
    kmeans = kmean.fit(imgvalues)
    finaldict[keyvalues] = kmeans.cluster_centers



with open('saved_dictionary_centroid.pkl', 'wb') as f:
    pickle.dump(imgdict, f)
    
with open('saved_dictionary_centroid.pkl', 'rb') as f:
    final_dict = pickle.load(f)

# CHECKPOINT_PATH='sam_vit_h_4b8939.pth'

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# MODEL_TYPE = "vit_h"


# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
# sam.eval()
# mask_generator = SamAutomaticMaskGenerator(sam)

# for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB'):
#     image = cv2.imread('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     output_mask = mask_generator.generate(image_rgb)
#     mask_annotator = sv.MaskAnnotator(opacity = 1, color_lookup = sv.ColorLookup.INDEX)
#     detections = sv.Detections.from_sam(output_mask)
#     annotated_image = mask_annotator.annotate(scene = image, detections = detections)
#     im = Image.fromarray(annotated_image)
#     im.save('datasets/landscape2photo/segmentation/'+i)


# # Generate segmentation mask
# output_mask = mask_generator.generate(image_rgb)



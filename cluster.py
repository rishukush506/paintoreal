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


segcolors = []
count = 0 
for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/segmentation'):
      colors, pixel_count = extcolors.extract_from_path("/home/ishika/kaushik/art2real/datasets/landscape2photo/segmentation/"+i)
      print(i)
      for j in range(len(colors)):
        segcolors.append(colors[j][0])

      if count > 10:
         break;
      count+=1    



segkeys = set(segcolors)
imgdict = {key: [] for key in segkeys}
print(imgdict)
print(len(segcolors))


unfold = nn.Unfold(kernel_size=(16, 16), stride = 6)
for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/segmentation'):
    imageseg = read_image('/home/ishika/kaushik/art2real/datasets/landscape2photo/segmentation/'+i)
    imageseg = imageseg.to(torch.float32)
    image = read_image('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
    image = image.to(torch.float32)

    outputseg = unfold(imageseg)
    outputimg = unfold(image)

    listseg =  outputseg.permute(1,0).tolist()
    listimg = outputimg.permute(1,0).tolist()
    cnt = 0
    for m in listseg:
        im = Image.fromarray((torch.Tensor(m).reshape(16, 16, 3).cpu().numpy()*255).astype(np.uint8))
        im.save("img.png")
        colors, pixel_count = extcolors.extract_from_path("img.png")
        for j in range(len(colors)):
              if (colors[j][1]/ pixel_count) > 0.25:
                   imgdict[colors[j][0]].append(listimg[cnt])

        cnt+=1


with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(imgdict, f)
        
with open('saved_dictionary.pkl', 'rb') as f:
    img_dict = pickle.load(f)

    

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



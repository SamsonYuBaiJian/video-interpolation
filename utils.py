# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from scipy import interpolate
# import cv2
# from PIL import Image
# import os
# from tqdm import tqdm

# class VimeoDataset():
#     def __init__(self, video_dir, text_split):
#         """
#         Args:
#             video_dir (string): Vimeo-90k sequences directory.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.video_dir = video_dir
#         self.text_split = text_split
#         self.first_last_frames = []

#         with open(self.text_split, 'r') as f:
#             filenames = f.readlines()
#             f.close()
#         final_filenames = []
#         self.save_filenames = []
#         #### create file directory sequece for the generated images
#         for i in filenames:
#             self.save_filenames.append(os.path.join('/home/hong/generated', i.split('\n')[0])) 
#             final_filenames.append(os.path.join(self.video_dir, i.split('\n')[0]))
#         #### create file directory sequece for the first frame and last frame in test dataset
#         for f in final_filenames:
            
#             try:
#                 frames = [os.path.join(f, i) for i in os.listdir(f)]
#             except:
#                 print("gg")
#                 break
#             frames = sorted(frames)
#             if len(frames) == 2:
#                 self.first_last_frames.append([frames[0], frames[1]])
        
#     def generate_results(self):
#         #print(len(self.first_last_frames))
#         for k in tqdm(range(len(self.first_last_frames))):
#             #load first frame
#             image1 = cv2.imread(self.first_last_frames[k][0])
#             #load last frame
#             image2 = cv2.imread(self.first_last_frames[k][1])
#             #split the rgb nxmx3 image array into their respective channel stacks
#             blue1, green1,red1 = cv2.split(image1)
#             blue2, green2,red2  = cv2.split(image2)
#             #as cv2 convention is bgr , rgb conversion is needed
#             img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#             img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#             n = blue1.shape[0]
#             m = blue1.shape[1]
#             out_blue = self.interpol(blue1,blue2,n,m)
#             out_green = self.interpol(green1,green2,n,m)
#             out_red = self.interpol(red1,red2,n,m)
#             arrays = [out_blue,out_green,out_red]
#             img = np.stack(arrays, axis=2)
#             dir = self.save_filenames[k]
#             if not os.path.exists(dir):
#                 os.makedirs(dir)
#             cv2.imwrite(os.path.join(self.save_filenames[k] , 'linear.png'),img)
            
            

#     def interpol(self,array1,array2,n,m):
#         #create the points defining the regular grid in n dimensions
#         points = (np.r_[0, 2], np.arange(n), np.arange(m))
#         # create the data on the regular grid in n dimensions
#         values = np.stack((array1, array2))
#         #create the  coordinates to sample the gridded data at
#         xi = np.rollaxis(np.mgrid[:n, :m], 0, 3).reshape((n*m, 2))
#         xi = np.c_[np.ones(n*m), xi]
#         #interpolate the stacked frames with linear method with nxm arrays
#         values_x = interpolate.interpn(points, values, xi, method='linear')
#         values_x = values_x.reshape((n, m))
#         values_x = values_x.astype(int)
#         return values_x
# #load your test set directory

# """Args:
#             video_dir (string): Vimeo-90k sequences directory.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#                 """
# dataset_dir = '/home/hong/Downloads/vimeo-90k/sequences'
# train_list =  '/home/hong/Downloads/vimeo-90k/tri_testlist.txt' 
# dataset = VimeoDataset(dataset_dir,train_list)
# dataset.generate_results()

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def change_pixel_intensities(image_path, beta=0):
    """
    Increases image pixel intensity/brightness.

    Args:
        image_path (string): Image file path.
        beta (int, optional): Pixel value increase.
    """
    img = Image.open(image_path)
    file_ex = image_path.split('.')[-1]
    img = np.array(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                img[y,x,c] = np.clip(img[y,x,c] + beta, 0, 255)

    img = Image.fromarray(img)
    img.save("./your_file.{}".format(file_ex))
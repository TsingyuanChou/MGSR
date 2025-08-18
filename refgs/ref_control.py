import cv2

import cv2
import numpy as np

# # Load the grayscale image
# ref_map = cv2.imread('results/tv_0.001smooth/test/ours_30000/ref_map/00003.png', cv2.IMREAD_GRAYSCALE)

# # Apply Otsu's thresholding
# _, mask = cv2.threshold(ref_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # mask = cv2.adaptiveThreshold(ref_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Load the color image
# image = cv2.imread('results/tv_0.001smooth/test/ours_30000/renders/00003.png')

# # Apply the binary mask to the color image
# scale = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1.1,1.2,1.3, 1.4, 1.5]
# for i in range(len(scale)):
#     enhanced_image = image.copy().astype(float) / 255.0
#     enhanced_image[mask == 0] *= scale[i] # Adjust the brightness of the corresponding regions
#     # Save the brightened image
#     cv2.imwrite('ref_control/tv_brightened_image{}.jpg'.format(scale[i]), (enhanced_image*255).astype(np.uint8))

# ref_map = cv2.imread('results/tv_0.001smooth/test/ours_30000/ref_map/00003.png', cv2.IMREAD_GRAYSCALE)
# print(ref_map.shape)
# ref_color = cv2.imread('results/tv_0.001smooth/test/ours_30000/ref_color/00003.png')
# print(ref_color.shape)
# trans_color = cv2.imread('results/tv_0.001smooth/test/ours_30000/trans_color/00003.png')
# image = cv2.imread('results/tv_0.001smooth/test/ours_30000/renders/00003.png')

# mask = cv2.imread('data/art1/refl_masks/IMG_9385.png')
# mask = cv2.resize(mask, (1296, 864), interpolation=cv2.INTER_AREA)

# ref_map = ref_map.astype(float) / 255.0
# ref_color = ref_color.astype(float) / 255.0
# trans_color = trans_color.astype(float) / 255.0
# image = image.astype(float) / 255.0
# # ref_map = ref_map[:, :, np.newaxis]
# scale = [0.5, 0.6,0.7, 0.8, 0.9, 1.0,1.1,1.2,1.3, 1.4, 1.5]
# # scale = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
# for i in range(len(scale)):
#     ref_copy = ref_color.copy()
#     # ref_copy[mask == 0] *= scale[i]
#     mask = ref_map > 0.5
#     ref_copy[mask] *= scale[i]
#     ref_copy = np.clip(ref_copy, 0, 1)
#     render = trans_color + ref_copy * ref_map[:, :, np.newaxis]
#     # combined_mask = ((mask == 255) & (image > 100)).astype(np.uint8) * 255
#     # render = image.copy()
#     # render[mask == 255] *= scale[i]
#     cv2.imwrite('ref_control/tv_render{}.jpg'.format(scale[i]), (render*255).astype(np.uint8))

# ref_map = cv2.imread('results/bookcase_0.001smooth/test/ours_30000/ref_map/00000.png', cv2.IMREAD_GRAYSCALE)
# ref_color = cv2.imread('results/bookcase_0.001smooth/test/ours_30000/ref_color/00000.png')
# trans_color = cv2.imread('results/bookcase_0.001smooth/test/ours_30000/trans_color/00000.png')
# ref_image = ref_color * ref_map[:, :, np.newaxis]
# img = trans_color + ref_image
# cv2.imwrite('ref_control/bookcase_00000_ref_image.jpg', (ref_image*255).astype(np.uint8))
# cv2.imwrite('ref_control/bookcase_00000_img.jpg', (img*255).astype(np.uint8))

ref_image = cv2.imread('art1_dsrnet_l_4000_test_r.png')
ref_image = ref_image.astype(float) / 255.0
scale = [1.0,1.2, 1.4, 1.6, 1.8, 2.0, 3.0]
for i in range(len(scale)):
    ref_copy = ref_image.copy()   
    ref_copy[ref_image > 0.3] *= scale[i]
    ref_copy = np.clip(ref_copy, 0, 1)
    cv2.imwrite('ref_control/art1_dsrnet_l_4000_test_r_{}.jpg'.format(scale[i]), (ref_copy*255).astype(np.uint8))

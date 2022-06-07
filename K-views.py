import os
import time
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def get_character_view(
    dataset, 
    init_centers,
    max_iter, 
    random_state,
):
    character_views_map = dict()
    # use kemans extract character views
    km = KMeans(n_clusters=init_centers.shape[0], 
                init=init_centers, 
                max_iter=max_iter, 
                random_state=random_state)
    km.fit(np.array([patch.flatten() / 255 for patch in dataset]))
    for label, c_view in enumerate(km.cluster_centers_):
        character_views_map[label] = c_view
    return character_views_map

def match_score(patch, character_views_map):
    min_score = np.inf
    pred_label = None
    flatten_patch = patch.flatten() / 255
    for label in character_views_map:
        for each_charater in character_views_map[label]:
            dist = np.linalg.norm(flatten_patch - each_charater)
            if min_score > dist:
                min_score = dist
                pred_label = label
    return min_score, pred_label

def predict(patch, character_views_map):
    min_dist = np.inf
    pred_label = None
    flatten_patch = patch.flatten() / 255
    for label in character_views_map:
        for each_charater in character_views_map[label]:
            dist = np.linalg.norm(flatten_patch - each_charater)
            if min_dist > dist:
                min_dist = dist
                pred_label = label
    return pred_label

def train(images, 
          init_centers, 
          max_iter, 
          random_state=42):
      character_view_map = get_character_view(images, init_centers, max_iter, random_state)
      return character_view_map
  
def image_classify(dataset, labels, character_views):
    statistics = dict()
    total_hit = 0    
    for single, label in zip(dataset, labels):
        _, pred_label = match_score(single, character_views)
        statistics[int(label)] = statistics.get(int(label), 0) + (pred_label == label)
        total_hit += pred_label == label
    for label, hit_num in statistics.items():
        print("label {} accuracy {}".format(label, hit_num / np.sum(labels == label)))
    print("all accuracy {}".format(total_hit / dataset.shape[0]))
    
def init_color_map(patch_size):
    color_map = {
        0: np.zeros((*patch_size, 3)), 
        1: np.zeros((*patch_size, 3)), 
        2: np.zeros((*patch_size, 3))
    }

    color_map[0][:, :, 0] = color_map[0][:, :, 0] + 255
    color_map[1][:, :, 1] = color_map[1][:, :, 1] + 255
    color_map[2][:, :, 2] = color_map[2][:, :, 2] + 255
    return color_map

def vis(image_list, 
        source_size, 
        patch_size, 
        row, 
        col, 
        color_map, 
        character_views_map):
    
    color_image_list = []
    idx = 0
    for r_i in range(row):
        for c_i in range(col):
            color_image = image_list[idx].copy()
            idx += 1
            image_col, image_row = source_size
            patch_col, patch_row = patch_size
            # col ...
            for i in range(0, image_col, patch_col):
                # row ...
                for j in range(0, image_row, patch_row):
                    # extract patch from image
                    patch = color_image[i:i + patch_col, j:j + patch_row, :]
                    # assign label to patch
                    label = predict(patch, character_views_map)
                    # apply label to pixel
                    color_image[i:i + patch_col, j:j + patch_row, :] = color_map[label]
            color_image_list.append(color_image)
            
    return color_image_list

def random_select_cluster_center(patch_list, label_list):
    label_idx_map = dict()
    for idx, label in enumerate(label_list):
        label_idx_map.setdefault(int(label), []).append(idx)
    
    init_center = []
    for label in range(len(label_idx_map)):
        # random choice
        init_center.append(patch_list[random.choice(label_idx_map[label])].flatten()  / 255)
    
    return np.array(init_center)

def mask_images(image_dir, 
                label, 
                source_size, 
                patch_size,
                ):
    
    patch_list = []
    label_list = []
    ori_img = []
    mask_img_list = []
    image_col, image_row = source_size
    patch_col, patch_row = patch_size
    ori_col, ori_row = 512, 512
    
    for img in os.listdir(image_dir):
        if not img.endswith('.jpg'):
            continue
        img_path = os.path.join(image_dir, img)
        prefix = '.'.join(img.split('.')[:-1])
        json_file = prefix + '.json'
        json_path = os.path.join(image_dir, json_file)
        with open(json_path, 'r') as f:
            data = f.read()
        
        data = json.loads(data)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (source_size[1], source_size[0]))
        h_ratio, w_ratio = img.shape[0] / ori_col, img.shape[1] / ori_row
        points = data["shapes"][0]["points"]
        new_point = []
        for point in points:
            new_point.append(
                [point[0] * w_ratio, point[1] * h_ratio]
            )
        points = np.array(new_point, np.int32)
        # convert BGR -> RBG
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_img.append(img)
        mask = np.zeros_like(img)
        mask_img_list.append(mask)
        cv2.fillPoly(mask, [points], (255, 255, 255))     
        # col ...
        for i in range(0, image_col - patch_col, patch_col):
            # row ...
            for j in range(0, image_row - patch_row, patch_row):
                # extract patch from image
                patch = mask[i:i + patch_col, j:j + patch_row, :].flatten()                            
                label_count = np.sum(patch == 255)
                # print(label_count)
                # if label_count == 0 or label_count == patch_col * patch_row:
                #     continue                
                # get label
                if 2 * label_count >= patch_col * patch_row:
                    cur_label = label
                else:
                    cur_label = 0
                ori_patch = img[i:i + patch_col, j:j + patch_row, :].flatten()     
                patch_list.append(ori_patch)
                label_list.append(cur_label)
    
    return ori_img, mask_img_list, patch_list, label_list

def vis_patch(row, col, patch_size, patch_list, patch_label, label_map):
    vis_image_list = []
    for label, title in label_map.items():
        all_stack = []
        vis = set()
        list_idx = list(range(len(patch_list)))
        for i in range(row):
            col_list = []
            for j in range(col):
                idx = random.choice(list_idx)
                while idx in vis or patch_label[idx] != label:
                    idx = random.choice(list_idx)
                vis.add(idx)
                col_list.append(patch_list[idx].reshape(*patch_size, 3))
            all_stack.append(col_list)
        divide_image = np.array(all_stack)
        m,n,grid_h, grid_w=[divide_image.shape[0],divide_image.shape[1],
                        divide_image.shape[2],divide_image.shape[3]]

        restore_image = np.zeros([m*grid_h, n*grid_w, 3], np.uint8)
        for i in range(m):
            for j in range(n):
                restore_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w]=divide_image[i,j,:]
                
        vis_image_list.append({
            'title': title, 
            'image': restore_image
        })
    return vis_image_list
    
if __name__ == '__main__':
    
    # source size and patch size
    source_size = (578, 578)
    patch_size = (17, 17)
    show_mask = False
    # kmeans params
    max_iter = 1000
    random_state = 42
    color_map = init_color_map(patch_size)
    # vis params
    row = 2
    col = 3
    patch_row = 10
    patch_col = 10
    figure_size = (24, 12)
    # label
    label = {
        0: 'background', 
        1: 'rough', 
        2: 'smooth'
    }
    
    rough_ori_img, rough_mask_img, rough_patch, rough_label = mask_images('./rough', 
                                                            1, 
                                                            source_size, 
                                                            patch_size, 
                                                            )

    
    
    smooth_ori_img, smooth_mask_img, smooth_patch, smooth_label = mask_images('./smooth', 
                                                            2, 
                                                            source_size, 
                                                            patch_size, 
                                                            )
    
    
    
    all_patch = np.array(rough_patch + smooth_patch)
    all_label = np.array(rough_label + smooth_label).reshape(-1, 1)
    image_list = rough_ori_img + smooth_ori_img
    mask_image_list = rough_mask_img + smooth_mask_img
    
    print('patch count: ')
    for c_l in label:
        print('current label {}'.format(label[c_l]))
        print('length: {}'.format(np.sum(all_label == c_l)))
    vis_img_list = vis_patch(patch_row, patch_col, patch_size, all_patch, all_label, label)
    
    if show_mask:
        plt.figure(figsize=figure_size)
        img_idx = 0    
        for _ in range(1, row + 1):
            for _ in range(1, col + 1):
                plt.subplot(row, col, img_idx + 1)
                plt.imshow(mask_image_list[img_idx])
                img_idx += 1
        plt.show()
        
    plt.figure(figsize=figure_size)
    img_idx = 0    
    for idx, item in enumerate(vis_img_list):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(item['image'])
        plt.title(item['title'])
        img_idx += 1
    plt.show()
    
    
    
    
    train_imgs, test_imgs, train_label, test_label = train_test_split(all_patch, 
                                                                      all_label, 
                                                                      test_size=0.1)
    print("train patch size {}".format(train_imgs.shape[0]))
    print("test patch size {}".format(test_imgs.shape[0]))
    # train ...
    init_centers = random_select_cluster_center(train_imgs, train_label)    
    character_views_map = train(train_imgs, init_centers, max_iter, random_state)
    print("Classification test set image...")
    image_classify(test_imgs, test_label, character_views_map)
    # vis
    color_image_list = vis(image_list, 
                            source_size, 
                            patch_size, 
                            row, 
                            col, 
                            color_map, 
                            character_views_map)
    plt.figure(figsize=figure_size)
    img_idx = 0    
    for r_i in range(1, row + 1):
        for c_i in range(1, col + 1):
            plt.subplot(row, 2 * col, (r_i - 1) * col + c_i + (r_i - 1) * col)
            plt.imshow(image_list[img_idx])
            img_idx += 1

    img_idx = 0
    for r_i in range(1, row + 1):
        for c_i in range(1, col + 1):
            plt.subplot(row, 2 * col, (r_i - 1) * col + c_i + r_i * col)
            im = plt.imshow(color_image_list[img_idx])
            img_idx += 1
            if c_i == col and r_i == 1:
                values = [0, 1, 2]
                colors = {
                    2: 'green', 
                    1: 'blue', 
                    0: 'red' 
                }
                patches = [ mpatches.Patch(color=colors[i], label="Label {l}".format(l=values[i]) ) for i in range(len(values)) ]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()
        
        
        
        
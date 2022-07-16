import enum
import os
import glob
import time
import json
import random

import cv2
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# seed_num = 1
# random.seed(seed_num)
# np.random.seed(seed_num)

def get_character_view(
    dataset, 
    init_centers,
    max_iter, 
    loss_threshold, 
    random_state, 
):
    character_views_map = dict()
    # use kemans extract character views
    km = KMeans(n_clusters=init_centers.shape[0], 
                init=init_centers, 
                max_iter=max_iter, 
                tol=loss_threshold, 
                random_state=random_state, 
                verbose=True)
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

def predict_new(image, 
                source_size, 
                patch_size, 
                character_views_map, 
                rough_threshold=0.5, 
                rgb_or_gray='gray'):
    image_col, image_row = source_size
    patch_col, patch_row = patch_size
    # col ...
    current_label = {
        1: 0, 
        2: 0
    }
    for i in range(0, image_col, patch_col):
        # row ...
        for j in range(0, image_row, patch_row):
            # extract patch from image
            patch = image[i:i + patch_col, j:j + patch_row, :]
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                continue
            if rgb_or_gray == 'gray':
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            # assign label to patch
            label = predict(patch, character_views_map)
            if label != 0:
                current_label[label] += 1
    rough_ratio = current_label[1] / (current_label[1] + current_label[2])
    return 'rough' if rough_ratio >= rough_threshold else 'smooth'          

def train(images, 
          init_centers, 
          max_iter, 
          loss_threshold, 
          random_state=42):
      character_view_map = get_character_view(images, init_centers, max_iter, loss_threshold, random_state)
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
        character_views_map, 
        rgb_or_gray,  
        rough_threshold=0.5):

    label_list = []
    color_image_list = []
    idx = 0
    for r_i in range(row):
        for c_i in range(col):
            color_image = image_list[idx].copy()
            idx += 1
            image_col, image_row = source_size
            patch_col, patch_row = patch_size
            # col ...
            current_label = {
                1: 0, 
                2: 0
            }
            for i in range(0, image_col, patch_col):
                # row ...
                for j in range(0, image_row, patch_row):
                    # extract patch from image
                    patch = color_image[i:i + patch_col, j:j + patch_row, :]
                    if rgb_or_gray == 'gray':
                        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    # assign label to patch
                    label = predict(patch, character_views_map)
                    # apply label to pixel
                    color_image[i:i + patch_col, j:j + patch_row, :] = color_map[label]
                    if label != 0:
                        current_label[label] += 1
            color_image_list.append(color_image)
            rough_ratio = current_label[1] / (current_label[1] + current_label[2])
            label_list.append(1 if rough_ratio >= rough_threshold else 2)            
    return color_image_list, label_list

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
                not_bg_num, 
                bg_num, 
                gaborparam, 
                rgb_or_gray, 
                An
                ):
    
    color_img = []
    ori_img = []
    mask_img_list = []
    image_col, image_row = source_size
    patch_col, patch_row = patch_size
    ori_col, ori_row = 512, 512
    bg_patch_list = []
    not_bg_patch_list = []
    
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
        #img = cv2.GaussianBlur(img, kernel_size, 0)
        gabor_img = process(img, build_filters(gaborparam))
        for i in range(An):
            # option 1. use absolute value
            gabor_img[i] = np.absolute(gabor_img[i] - img)
            gabor_img[i] = (1 - gabor_img[i] / np.max(gabor_img[i].flatten())) * 255
        img = gabor_img[0]
        color_img.append(img.copy())
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_img_list.append(mask)
        if rgb_or_gray == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ori_img.append(img)
        # col ...
        for i in range(0, image_col - patch_col, patch_col):
            # row ...
            for j in range(0, image_row - patch_row, patch_row):
                # extract patch from image
                patch = mask[i:i + patch_col, j:j + patch_row].flatten()                            
                label_count = np.sum(patch == 255)
                # print(label_count)
                if label_count != 0 and label_count != patch_col * patch_row:
                    continue
                ori_patch = img[i:i + patch_col, j:j + patch_row].flatten()
                if 2 * label_count >= patch_col * patch_row:
                    not_bg_patch_list.append(ori_patch)
                else:
                    bg_patch_list.append(ori_patch)
    patch_list = []
    label_list = []
    if not_bg_num < len(not_bg_patch_list):
        patch_list.extend(random.sample(not_bg_patch_list, not_bg_num))
    else:
        patch_list.extend(not_bg_patch_list)
    if bg_num < len(bg_patch_list):
        patch_list.extend(random.sample(bg_patch_list, bg_num))
    else:
        patch_list.extend(bg_patch_list)
    label_list.extend([label] * min(not_bg_num, len(not_bg_patch_list)) +\
        [0] * min(bg_num, len(bg_patch_list)))
    return color_img, ori_img, mask_img_list, patch_list, label_list

def vis_patch(row, 
              col, 
              patch_size, 
              patch_list, 
              patch_label, 
              label_map, 
              rgb_or_gray):
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
                if rgb_or_gray == 'gray':
                    col_list.append(patch_list[idx].reshape(*patch_size))
                else:
                    col_list.append(patch_list[idx].reshape((*patch_size, 3)))
            all_stack.append(col_list)
        divide_image = np.array(all_stack)
        m, n, grid_h, grid_w=[divide_image.shape[0], 
                            divide_image.shape[1],
                            divide_image.shape[2], 
                            divide_image.shape[3]]
        if rgb_or_gray == 'gray':
            restore_image = np.zeros([m * grid_h, n * grid_w], np.uint8)
        else:
            restore_image = np.zeros([m * grid_h, n * grid_w, 3], np.uint8)
        for i in range(m):
            for j in range(n):
                if rgb_or_gray == 'gray':
                    restore_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w] = divide_image[i, j]
                else:
                    restore_image[i * grid_h : (i+1) * grid_h, 
                                  j * grid_w : (j+1) * grid_w, :] = divide_image[i, j]
                
        vis_image_list.append({
            'title': title, 
            'image': restore_image
        })
    return vis_image_list

def build_filters(gaborparam):
    THETAS=gaborparam[0] 
    SIGMA=gaborparam[1] 
    LAMBDA=gaborparam[2] 
    GAMMA=gaborparam[3] 
    PSI=gaborparam[4]
    KERNEL_SIZE=gaborparam[5]
    
    filters = []
    for theta in THETAS:
        kern = cv2.getGaborKernel(
                ksize=KERNEL_SIZE,
                sigma=SIGMA,
                theta=theta,
                lambd=LAMBDA,
                gamma=GAMMA,
                psi=PSI, 
                ktype=cv2.CV_32F,
        )
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def convolve(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def process(img, filters):
    res = []
    for filter in filters:
        res.append(convolve(img, [filter]))
    return np.array(res)

def gen_gaborparam(An=4, 
                   SIGMA = 1, 
                   LAMBDA = np.pi / 10, 
                   GAMMA = 0.2, 
                   PSI = 1.0, 
                   KERNEL_SIZE=(5,5)):
    
    THETAS = np.arange(0, np.pi, np.pi / An).tolist()
    gaborparam = THETAS, SIGMA, LAMBDA, GAMMA, PSI, KERNEL_SIZE
    return gaborparam

if __name__ == '__main__':
    
    # source size and patch size
    source_size = (450, 450)
    patch_size = (15, 15)
    show_mask = False
    # kmeans params
    max_iter = 10000
    loss_threshold = 1e-9
    random_state = 42
    # rgb / gray
    rgb_or_gray = 'gray'
    color_map = init_color_map(patch_size)
    # vis params
    row = 2
    col = 2
    patch_row = 10
    patch_col = 10
    figure_size = (24, 12)
    kernel_size = (3, 3)
    # label
    label = {
        0: 'background', 
        1: 'rough', 
        2: 'smooth'
    }
    # sample label
    rough_patch_num = 100000
    smooth_patch_num = 100000
    bg_patch_num = 100000
    An = 4
    gaborparam = gen_gaborparam(An=An)
    
    
    color_rought_img, rough_ori_img, rough_mask_img, rough_patch, rough_label = mask_images('./rough', 
                                                                                            1, 
                                                                                            source_size, 
                                                                                            patch_size, 
                                                                                            rough_patch_num, 
                                                                                            bg_patch_num // 2 if bg_patch_num % 2 == 0 else (bg_patch_num // 2 + 1), 
                                                                                            gaborparam, 
                                                                                            rgb_or_gray, 
                                                                                            An
                                                                                            )

    
    
    color_smooth_img, smooth_ori_img, smooth_mask_img, smooth_patch, smooth_label = mask_images('./smooth', 
                                                                                                2, 
                                                                                                source_size, 
                                                                                                patch_size, 
                                                                                                smooth_patch_num, 
                                                                                                bg_patch_num // 2, 
                                                                                                gaborparam, 
                                                                                                rgb_or_gray, 
                                                                                                An
                                                                                                )
                                        
    
    all_patch = np.array(rough_patch + smooth_patch)
    all_label = np.array(rough_label + smooth_label).reshape(-1, 1)
    gray_image_list = rough_ori_img + smooth_ori_img
    ori_image_list = color_rought_img + color_smooth_img
    mask_image_list = rough_mask_img + smooth_mask_img
    
    
    print('patch count: ')
    for c_l in label:
        print('current label {}'.format(label[c_l]))
        print('length: {}'.format(np.sum(all_label == c_l)))
    # vis_img_list = vis_patch(patch_row, 
    #                          patch_col, 
    #                          patch_size, 
    #                          all_patch, 
    #                          all_label, 
    #                          label, 
    #                          rgb_or_gray)
    
    # if show_mask:
    #     plt.figure(figsize=figure_size)
    #     img_idx = 0    
    #     for _ in range(1, row + 1):
    #         for _ in range(1, col + 1):
    #             plt.subplot(row, col, img_idx + 1)
    #             plt.imshow(mask_image_list[img_idx], 
    #                        cmap=plt.get_cmap('gray'))
    #             img_idx += 1
    #     plt.show()
        
    # plt.figure(figsize=figure_size)
    # img_idx = 0    
    # for idx, item in enumerate(vis_img_list):
    #     plt.subplot(1, 3, idx + 1)
    #     plt.imshow(item['image'], 
    #                cmap=plt.get_cmap('gray') if rgb_or_gray == 'gray' else None)
    #     plt.title(item['title'])
    #     img_idx += 1
    # plt.show()
    
    train_imgs, test_imgs, train_label, test_label = train_test_split(all_patch, 
                                                                      all_label, 
                                                                      test_size=0.1)
    print("train patch size {}".format(train_imgs.shape[0]))
    print("test patch size {}".format(test_imgs.shape[0]))
    # train ...
    init_centers = random_select_cluster_center(train_imgs, train_label)    
    character_views_map = train(train_imgs, 
                                init_centers, 
                                max_iter, 
                                loss_threshold, 
                                random_state)
    print("Classification test set image...")
    image_classify(test_imgs, test_label, character_views_map)
    
    # vis
    color_image_list, label_list = vis(ori_image_list, 
                                        source_size, 
                                        patch_size, 
                                        row, 
                                        col, 
                                        color_map, 
                                        character_views_map, 
                                        rgb_or_gray)
    # plt.figure(figsize=figure_size)
    # img_idx = 0    
    # for r_i in range(1, row + 1):
    #     for c_i in range(1, col + 1):
    #         plt.subplot(row, 2 * col, (r_i - 1) * col + c_i + (r_i - 1) * col)
    #         plt.imshow(
    #             ori_image_list[img_idx] if rgb_or_gray == 'rgb' \
    #                 else cv2.cvtColor(ori_image_list[img_idx], cv2.COLOR_RGB2GRAY), 
    #             cmap=plt.get_cmap('gray') if rgb_or_gray == 'gray' else None
    #             )
    #         img_idx += 1

    # img_idx = 0
    # for r_i in range(1, row + 1):
    #     for c_i in range(1, col + 1):
    #         plt.subplot(row, 2 * col, (r_i - 1) * col + c_i + r_i * col)
    #         im = plt.imshow(color_image_list[img_idx], 
    #                         cmap=plt.get_cmap('gray') if rgb_or_gray == 'gray' else None)
    #         img_idx += 1
    #         if c_i == col and r_i == 1:
    #             values = [0, 1, 2]
    #             colors = {
    #                 2: 'green', 
    #                 1: 'blue', 
    #                 0: 'red' 
    #             }
    #             label_name = {
    #                 0: 'background', 
    #                 1: 'rough', 
    #                 2: 'smooth'
    #             }
                
    #             patches = [ mpatches.Patch(color=colors[i], label="Label {}".format(label_name[i]) ) for i in range(len(values)) ]
    #             plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # plt.show()
    
    # # show labels    
    plt.figure(figsize=figure_size)
    img_idx = 0 
    for r_i in range(1, row + 1):
        for c_i in range(1, col + 1):
            plt.subplot(row, 2 * col, img_idx + 1)
            plt.imshow(
                ori_image_list[img_idx] if rgb_or_gray == 'rgb' \
                    else cv2.cvtColor(ori_image_list[img_idx], cv2.COLOR_RGB2GRAY), 
                cmap=plt.get_cmap('gray') if rgb_or_gray == 'gray' else None)
            plt.title('rough' if label_list[img_idx] == 1 else 'smooth')
            img_idx += 1
    plt.show()
    
    plt.figure(figsize=(20, 20))
    plt.title('prediction number {}'.format(20))
    predict_number = 20
    row = 4
    col = 5
    img_idx = 1
    for img_path in random.sample(glob.glob('./dataset/data/*.jpg'), predict_number):
        img = cv2.imread(img_path)
        print('current image {}'.format(img_path))
        label = predict_new(img, source_size, patch_size, character_views_map, rgb_or_gray)
        print('current label {}'.format(label))
        plt.subplot(row, col, img_idx)
        plt.imshow(                
                   ori_image_list[img_idx] if rgb_or_gray == 'rgb' \
                    else cv2.cvtColor(ori_image_list[img_idx], cv2.COLOR_RGB2GRAY), 
                   cmap=plt.get_cmap('gray'))
        plt.title(label)
        img_idx += 1
        
    plt.show()
    image_file = glob.glob('./dataset/data/*.jpg')
    
    label_map = {
        'smooth': 1, 
        'rough': 0
    }
    
    for csv_file in glob.glob('./data/*.csv'):
        # chose file
        if 'kvresults25_20.csv' not in csv_file:
            continue        
        csv_data = pandas.read_csv(csv_file)
        image_id = csv_data.id
        truth_label = csv_data.label
        hit = 0
        csv_name = csv_file.split("/")[-1]        
        for idx, (_id, t_label) in enumerate(zip(image_id, truth_label)):
            image_path = './dataset/data/{}.jpg'.format(_id)
            print('current image {}'.format(image_path))
            img = cv2.imread(image_path)
            gabor_img = process(img, build_filters(gaborparam))
            for i in range(An):
                gabor_img[i] = np.absolute(gabor_img[i] - img)
                gabor_img[i] = (1 - gabor_img[i] / np.max(gabor_img[i].flatten())) * 255
            img = gabor_img[0]
            label = predict_new(img, 
                                source_size, 
                                patch_size, 
                                character_views_map, 
                                0.5, 
                                rgb_or_gray)
            hit += (t_label == label_map[label])
            csv_data.predict[idx] = label_map[label]
        print(csv_name, 'acc', hit / len(csv_data))
        csv_data.to_csv(csv_name)
        
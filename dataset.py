import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL                import Image
from torch.nn           import functional as F
from torch.utils.data   import Dataset

class Edge_generator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = image.clone().float()
        shape = image.shape
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        img = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff.numpy()
        if return_all :
            return diff, img, erosion
        else:
            return diff
    
    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)

class MyDataset(Dataset):
    
    def read_img_path(self, img_path, test_fold, t_or_v_or_t):
        img_list = []
        img_ = sorted(os.listdir(img_path))

        for file_name in img_:
            parts = file_name.split("_")
            number_part = parts[0]
            label_part = parts[1]

            if t_or_v_or_t == 'train':
                if int(number_part) not in test_fold and label_part == 'train':
                    img_list.append(os.path.join(img_path, file_name))
            elif t_or_v_or_t == 'val':
                if int(number_part) not in test_fold and label_part == 'val':
                    img_list.append(os.path.join(img_path, file_name))
            elif t_or_v_or_t == 'test':
                if int(number_part) in test_fold:
                    img_list.append(os.path.join(img_path, file_name))

        return img_list
    
    # def map_colors_to_labels(self, image):
    #     color_mapping = {(64, 140, 216) : 0,   
    #                      (0, 255, 0)    : 1,   
    #                      (0, 0, 255)    : 2, 
    #                      (255, 0, 0)    : 3}
        
    #     label_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    #     for y in range(image.shape[0]):
    #         for x in range(image.shape[1]):
    #             pixel_color = tuple(image[y, x])
    #             if pixel_color in color_mapping:
    #                 label_image[y, x] = color_mapping[pixel_color]
    #             else:
    #                 print('标签中存在别的颜色！')
    #     return label_image
            
    def get_list(self):
        return self.img_list_1, self.img_list_2, self.label_list
    
    def __init__(self, img_path_1, img_path_2, label_path, test_fold, has_edge, train_val_test):
        super().__init__()
        self.img_path_1 = img_path_1
        self.img_path_2 = img_path_2
        self.label_path = label_path
        
        self.img_list_1 = self.read_img_path(self.img_path_1, test_fold, train_val_test)
        self.img_list_2 = self.read_img_path(self.img_path_2, test_fold, train_val_test)
        self.label_list = self.read_img_path(self.label_path, test_fold, train_val_test)

        self.edge = has_edge
        # print(self.edge)
        self.edge_generator =  Edge_generator(kernel_size=3)
        self.tvt = train_val_test
 
    def __getitem__(self, index): #双模态
        
        img_1 = cv2.imread(self.img_list_1[index])
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

        img_2 = cv2.imread(self.img_list_2[index])
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        # print(img_2.shape)

        label = cv2.imread(self.label_list[index], cv2.IMREAD_GRAYSCALE)
        # print(label.shape)

        if self.tvt == 'train':
            img_1, img_2, label = train_tf(img_1, img_2, label)
            img_1 = to_tensor(img_1)
            img_2 = to_tensor(img_2)
            img = torch.concat([img_1,img_2], axis=0)
            label = torch.from_numpy(label)
            if self.edge:
                edge1 = self.edge_generator((label==0) * 1.0)[0]
                edge2 = self.edge_generator((label==1) * 1.0)[0]
                edge3 = self.edge_generator((label==2) * 1.0)[0]
                edge4 = self.edge_generator((label==3) * 1.0)[0]
                edge = torch.from_numpy(np.concatenate((edge1, edge2, edge3, edge4), axis=0))
                return img, label, edge
            else:
                return img, label
        elif self.tvt == 'val' or self.tvt == 'test':
            img_1 = to_tensor(img_1)
            img_2 = to_tensor(img_2)
            img = torch.concat([img_1,img_2], axis=0)
            label = torch.from_numpy(label)
            if self.edge:
                edge1 = self.edge_generator((label==0) * 1.0)[0]
                edge2 = self.edge_generator((label==1) * 1.0)[0]
                edge3 = self.edge_generator((label==2) * 1.0)[0]
                edge4 = self.edge_generator((label==3) * 1.0)[0]
                edge = torch.from_numpy(np.concatenate((edge1, edge2, edge3, edge4), axis=0))
                return img, label, edge
            else:
                return img, label

    def __len__(self):
        return len(self.label_list)

def train_tf(im1, im2, la):
    p = np.random.choice([0, 1])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        ])
    im1 = train_tf(Image.fromarray(im1))
    im2 = train_tf(Image.fromarray(im2))
    la = train_tf(Image.fromarray(la))
    return np.array(im1), np.array(im2), np.array(la)

def to_tensor(data):
    to_tensor = transforms.Compose([
        transforms.ToTensor()])
    data = to_tensor(data)
    return data
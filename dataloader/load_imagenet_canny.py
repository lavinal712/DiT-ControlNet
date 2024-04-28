from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os


class CannyDetection(object):
    def __call__(self, image):
        image = np.array(image)
        edges = cv2.Canny(image, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb


class ImageNetCanny(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.edge_transform = CannyDetection()

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image_original = self.transform(image)

        image_edges = self.edge_transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image_original, image_edges, target

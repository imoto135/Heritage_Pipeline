
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
from torchvision.transforms import transforms
from natsort import natsorted

class SingleDataset(Dataset,):
    def __init__(self, path, imsize, train=True):
        # self.files = []
        # print(len(path))
        # for i in range(len(path)):
        #   self.files += sorted(glob.glob(path[i]))[:500]
        self.files = natsorted(glob.glob(path))
        self.imsize = imsize
        self.train = train
        
        print('DDRM Data Len:{}'.format(len(self.files)))
    def __getitem__(self, index):
        fname = self.files[index]
        print(fname)
        img = Image.open(fname).convert('RGB')
        # label_A = torch.tensor(int(fnameA[-6]))
        transform = self.__transform()
        item = transform(img)
        classes = 0
        # item_row = transform_row(img_A)
        return [item,classes]
    
    def __len__(self):
        return len(self.files)
    
    def __transform(self,):
        list = []
        if self.train:
            list += [
                     transforms.Resize((int(self.imsize*1.12),int(self.imsize*1.12)), transforms.InterpolationMode("bicubic")),
                     transforms.RandomCrop(self.imsize),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] に正規化（学習と一致）
                     ]
        else:
            list += [
                     transforms.Resize((int(self.imsize),int(self.imsize)), transforms.InterpolationMode("bicubic")),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] に正規化（学習と一致）
                     ]
        return transforms.Compose(list)
    
    
class PairDataset(Dataset):
    def __init__(self, pathA, pathB, imsize, isAlign=False):
       self.filesA = sorted(glob.glob(pathA))    #test 2000:
       self.filesB = sorted(glob.glob(pathB))    #test 4000:
       self.isAlign = isAlign
       self.imgsize = imsize
    def __getitem__(self, index):
        fnameA = self.filesA[index]
        img_A = Image.open(fnameA)
        # label_A = torch.tensor(int(fnameA[-6]))
        if self.isAlign==False:
            fnameB = self.filesB[random.randint(0, len(self.filesB)-1)]
            img_B = Image.open(fnameB)
        else:
            fnameB = self.filesB[index]
            img_B = Image.open(fnameB)
        # label_B = torch.tensor(int(fnameB[-6]))
        transform = self.__transform()
        item_A = transform(img_A)
        item_B = transform(img_B)
        # item_row = transform_row(img_A)
        return {'A':item_A, 'B':item_B, }
    
    def __len__(self):
        return len(self.filesA)
    
    def __transform(self,):
        list = []
        list += [
                 transforms.Resize(int(self.imsize*1.12), transforms.InterpolationMode("bicubic")),
                 transforms.RandomCrop(self.imsize),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),   
                 transforms.Lambda(lambda t:2*t-1)]
        return transforms.Compose(list)
    

from PIL import Image
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
import torch
from torchvision import transforms as T
import os
import random
import numpy as np
from numpy import repeat
from itertools import chain

import soundfile as sf
from torch import Tensor
import glob
#
# torch.multiprocessing.set_sharing_strategy('file_system')


trans = {300:T.Compose([T.Resize(300), T.ToTensor()]),
         128:T.Compose([T.Resize((128, 128)), T.ToTensor()]),
         299:T.Compose([T.ToTensor(), T.Resize(299)]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256))]),
         224:T.Compose([T.ToTensor(), T.Resize((224, 224))]),
         192:T.Compose([T.ToTensor(),T.Resize((192, 192))])}

def get_csv_path(root, train, DIL=False, reverse=False):
    pattern = "*_train.csv" if train else "*_test.csv"
    if DIL:
        csv_path = glob.glob(os.path.join(root, pattern))
        csv_path = sorted(csv_path)#[::-1]
        if reverse:
            csv_path = csv_path[::-1]
    else:
        csv_path = str(os.path.join(root, pattern))
    return csv_path

def count_png_files(dir_path):
    png_count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.png'):
                png_count += 1
    return png_count

class Multimodal_dataset(Dataset):
    def __init__(self, root, train, base_ratio, num_phases, augment, inplace_repeat, shuffle_seed, image_size, csv_dir=None, num_frame=4, num_classes=4, IL_batch_size=16, num_workers=4, reverse=False):
        csv_paths = get_csv_path(csv_dir, train, DIL=True, reverse=reverse)
        self.num_domains = len(csv_paths)
        self.num_classes = num_classes
        self.base_ratio = base_ratio
        self.inplace_repeat = inplace_repeat
        self.root = root
        self.domain_indices = [[] for _ in range(self.num_domains)]
        self.raw_dataset_length = 0
        assert image_size in trans.keys()
        
        self.IL_batch_size = IL_batch_size
        self.num_workers = num_workers

        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        self.img_aud = []
        domain_idx = 0
        i = 0
        for csv_path in csv_paths:
            fh = open(csv_path, 'r')
            img_aud = []
            line_num = 0
            for line in fh:
                # multidataset
                if len(line.split(' ')) == 1:
                    if line_num == 0:
                        # The first line is the root path of the dataset
                        self.root = line.strip()
                        continue

                line = line.split(' ')
                img_path = os.path.join(self.root, line[0])

                # 获取音频文件
                basename = os.path.basename(img_path)
                aud_path = os.path.join(self.root, img_path, basename + ".wav")
                if not os.path.exists(aud_path):
                    continue
                duration = count_png_files(img_path)
                if duration < 100:
                    # 少于100个视频的类别不考虑
                    continue
                
                img_aud.append((img_path, aud_path, int(line[-2]), int(line[-1])))
                
                self.domain_indices[domain_idx].append(i)
                i += 1
                line_num += 1
            print(csv_path, len(img_aud))
            self.img_aud.extend(img_aud)
            domain_idx += 1
        self.original_dataset_length = len(self.img_aud) # to mark the length of the original dataset excluding the memory
        self.trans = trans[image_size]
        self.image_size = image_size
        self.num_frame = num_frame
        self.base_size = int(self.num_domains * self.base_ratio)
        self.incremental_size = self.num_domains - self.base_size
        self.phase_size = self.incremental_size // (num_phases-1) if num_phases > 0 else 0
        print(f'num_domains: {self.num_domains}, base_size: {self.base_size}, incremental_size: {self.incremental_size}, phase_size: {self.phase_size}, num_classes: {self.num_classes}')

    def __getitem__(self, index):
        # img_data = torch.zeros((self.num_frame*10, 3, self.image_size, self.image_size))
        img_data = []
        fn_img, fn_aud, label, _ = self.img_aud[index]
        # temp = os.path.split(fn_img)[0]
        filename = temp = fn_img
        # index = int(os.path.split(fn_img)[1][:-4])
        temp1 = ''
        # pp = True

        ## 随机每10帧内取4帧
        slice_index = np.arange(0, 10, 1)
        random.shuffle(slice_index)
        slice_index = slice_index[:self.num_frame]
        slice_index.sort()
        slice_index = slice_index.repeat(10)
        slice_index = slice_index.reshape(self.num_frame, 10).transpose(1, 0)
        a = np.arange(0, 100, 10).reshape(10, 1)
        slice_index = slice_index + a
        # print(slice_index)
        slice_index = slice_index.reshape(-1)

        base = 0
        for i in range(len(slice_index)):

            fn = temp + '/' + str(slice_index[i] + base).zfill(5) + '.png'

            while i == 0 and not os.path.exists(fn):
                base+=1
                fn = temp + '/' + str(slice_index[i] + base).zfill(5) + '.png'
            try:
                img = Image.open(fn).convert('RGB')
                img = self.trans(img)
                img_data.append(img.unsqueeze(0))
                temp1 = fn
            except:
                print(filename+'.'*10)
        img_data = torch.cat(img_data, dim=0)
        img_data = img_data.view(-1, self.image_size, self.image_size)
        start = 0
        aud_data, _ = sf.read(fn_aud, start=start*16000, stop=(start+4)*16000)
        if len(aud_data.shape) == 2:
            aud_data = aud_data[:,0]
        aud_data = Tensor(aud_data)
        if aud_data.size(0) < 16000*4:
            aud_data = torch.cat([aud_data, torch.zeros(16000*4-aud_data.size(0))], dim=0)

        if label == 0:
            video_label = 0
            audio_label = 0
            total_label = 0
        elif label == 1:
            video_label = 1
            audio_label = 1
            total_label = 1
        elif label == 2 :
            video_label = 1
            audio_label = 0
            total_label = 2
        else:
            video_label = 0
            audio_label = 1
            total_label = 3
        if self.num_classes == 2:
            total_label = 1 if total_label > 0 else 0

        return index, (img_data, aud_data), (video_label, audio_label, total_label), (fn_img, fn_aud, label, start)

    def __len__(self):
        return len(self.img_aud)
    
    def _subset(self, label_begin: int, label_end: int) -> Subset[T_co]:
        # print(f'label_begin, label_end: {label_begin, label_end}')
        ids = self.domain_indices[label_begin:label_end]
        if self.__len__() > self.original_dataset_length:
            # added memory
            ids.append(list(range(self.original_dataset_length, self.__len__())))
        sub_ids = tuple(chain.from_iterable(ids))
        return Subset(self, repeat(sub_ids, self.inplace_repeat).tolist())

    def subset_at_phase(self, phase: int, memory: [] = []) -> Subset[T_co]:
        self.reset()
        if len(memory)>0:
            self.append_memory(memory)
        if phase == 0:
            return self._subset(0, self.base_size)
        return self._subset(
            self.base_size + (phase - 1) * self.phase_size,
            self.base_size + phase * self.phase_size,
        )

    def subset_until_phase(self, phase: int) -> Subset[T_co]:
        return self._subset(
            0,
            self.base_size + phase * self.phase_size,
        )

    def append_memory(self, memory) -> None:
        self.img_aud.extend(memory)

    def reset(self) -> None:
        # delete the memory
        self.img_aud = self.img_aud[:self.original_dataset_length]

if __name__ == '__main__':
    image_size = 128
    frame_num = 4
    train_dataset = Multimodal_dataset(image_size, 'train', '/mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces/train.txt', num_frame=frame_num)
    test_dataset = Multimodal_dataset(image_size, 'test', '/mnt/200ssddata2t/yejianbin/FakeAVCeleb/cropped_faces/test.txt', num_frame=frame_num)

    print(train_dataset[0])
    print(test_dataset[0])
    print(train_dataset.subset_at_phase(0)[0])
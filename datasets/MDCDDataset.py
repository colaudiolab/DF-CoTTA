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
import random
trans = {300:T.Compose([T.Resize(300), T.ToTensor()]),
         128:T.Compose([T.Resize((128, 128)), T.ToTensor()]),
         299:T.Compose([T.ToTensor(), T.Resize(299)]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256))]),
         224:T.Compose([T.ToTensor(), T.Resize((224, 224))]),
         192:T.Compose([T.ToTensor(),T.Resize((192, 192))])}

def get_csv_path(root, train, DIL=False, order=''):
    pattern = "*_train.csv" if train else "*_test.csv"
    if DIL:
        csv_path = glob.glob(os.path.join(root, pattern))
        csv_path = sorted(csv_path)#[::-1]
        if order == 'reverse':
            csv_path = csv_path[::-1]
        elif order == 'random':
            rng = random.Random(0)
            rng.shuffle(csv_path)
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

class MDCDDataset(Dataset):
    def __init__(self, root, train, base_ratio, num_phases, inplace_repeat, image_size, csv_dir=None, num_frame=4, num_classes=4, IL_batch_size=16, num_workers=4, order='', CL_type='DIL', shot=10):
        self.CL_type = CL_type
        csv_paths = get_csv_path(csv_dir, train, DIL=True, order=order)
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
            real_num=0
            fake_num = 0
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
                label = int(line[-2])
                if label > 0:
                    fake_num+=1
                else:
                    real_num+=1
                img_aud.append((img_path, aud_path, label, int(line[-1])))
                
                self.domain_indices[domain_idx].append(i)
                i += 1
                line_num += 1
            # if domain_idx > 0 and train:
            #     # few shot
            #     img_aud, label_list = self.select_sample(img_aud, domain_idx, num=shot)
            #     real_num = label_list[0]
            #     fake_num = sum(label_list.values())-real_num
            print(csv_path, len(img_aud))
            # print(f'fake: {fake_num}, real: {real_num}')
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
        self.label_map = {}

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

        if self.CL_type == 'CIL':
            # if label > 0:
            #     if str(label) not in self.label_map:
            #         # this label first occur, then establish label mapping
            #         print(self.label_map)
            #         self.label_map[str(label)] = len(self.label_map) + 1
            #     total_label = video_label = audio_label = self.label_map[str(label)]
            # else:
            total_label = label
            video_label = label
            audio_label = label
        elif self.CL_type == 'DIL' or self.CL_type == 'TIL':
            if label == 0:
                video_label = 0
                audio_label = 0
                total_label = 0
            elif label == 1:
                video_label = 1
                audio_label = 1
                total_label = 3
            elif label == 2 :
                video_label = 1
                audio_label = 0
                total_label = 2
            else:
                video_label = 0
                audio_label = 1
                total_label = 1 
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
            # ids.append(list(range(self.original_dataset_length, self.__len__())))
            ids.extend(list(range(self.original_dataset_length, self.__len__())))
        ids = list(chain.from_iterable(ids))
        # sub_ids = tuple(chain.from_iterable(ids))
        # return Subset(self, repeat(sub_ids, self.inplace_repeat).tolist())
        return tmpDataset(self.img_aud, ids, self.image_size, self.num_frame, self.num_classes, self.IL_batch_size, self.num_workers, self.CL_type)

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

    def select_sample(self, img_aud, domain_idx, num=50):
        random.shuffle(img_aud)
        selected_img_aud = []
        label_list = {}
        for sample in img_aud:
            _, _, label, _ = sample
            if label not in label_list or label_list[label] < num:
                selected_img_aud.append(sample)
                if label not in label_list:
                    label_list[label] = 1
                else:
                    label_list[label] += 1
            elif label == 0 and label_list[label] < (len(label_list)-1)*num:
                selected_img_aud.append(sample)
                label_list[label] += 1
            
            if len(label_list) >= 4 and sum(label_list.values()) >= 6*num:
                break
        total_num = sum(label_list.values())
        if domain_idx == 0:
            self.domain_indices[domain_idx][:total_num]
        else:
            start_idx = self.domain_indices[domain_idx-1][-1] + 1
            end_idx = start_idx + total_num
            self.domain_indices[domain_idx] = list(range(start_idx, end_idx))
        return selected_img_aud, label_list

class tmpDataset(Dataset):
    def __init__(self, original_img_aud, indices, image_size=128, num_frame=4, num_classes=2, IL_batch_size=16, num_workers=4, CL_type='TIL'):
        self.CL_type = CL_type
        self.num_classes = num_classes
                
        self.IL_batch_size = IL_batch_size
        self.num_workers = num_workers

        assert image_size in trans.keys()
        self.trans = trans[image_size]
        self.image_size = image_size
        self.num_frame = num_frame
        self.img_aud = []
        # indices = indices[0] if isinstance(indices[0], list)  else indices
        for idx in indices:
            self.img_aud.append(original_img_aud[idx])

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

        if self.CL_type == 'CIL':
            total_label = label
            video_label = label
            audio_label = label
        elif self.CL_type == 'DIL' or self.CL_type == 'TIL':
            if label == 0:
                video_label = 0
                audio_label = 0
                total_label = 0
            elif label == 1:
                video_label = 1
                audio_label = 1
                total_label = 3
            elif label == 2 :
                video_label = 1
                audio_label = 0
                total_label = 2
            else:
                video_label = 0
                audio_label = 1
                total_label = 1 
            if self.num_classes == 2:
                total_label = 1 if total_label > 0 else 0

        return index, (img_data, aud_data), (video_label, audio_label, total_label), (fn_img, fn_aud, label, start)

    def __len__(self):
        return len(self.img_aud)
    
    def subset(self, ids):
        return tmpDataset(self.img_aud, ids, self.image_size, self.num_frame, self.num_classes, self.IL_batch_size, self.num_workers, self.CL_type)

    def removeSamples(self, indices):
        new_img_aud = [self.img_aud[idx] for idx in range(len(self.img_aud)) if idx not in indices]
        self.img_aud = new_img_aud

if __name__ == '__main__':
    image_size = 128
    frame_num = 4
    train_dataset = MDCDDataset(root='', train=True, base_ratio=0.2, num_phases=6, inplace_repeat=1, image_size=image_size, csv_dir='/mnt/200ssddata2t/yejianbin/MDCD-DB/Sequetail_incremental_DB/Task_incremental_DB/cross_dataset', reverse=True)
    # test_dataset = MDCDDataset(image_size, 'test', '/mnt/200ssddata2t/yejianbin/MDCD-DB/Sequetail_incremental_DB/Task_incremental_DB/cross_dataset/6AV1M_test.csv', num_frame=frame_num)

    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=True
    )
    from tqdm import tqdm
    for i in tqdm(data_loader):
        idx, data, label, fn = i
        pass
    # print(train_dataset[0])
    # print(test_dataset[0])
    # print(train_dataset.subset_at_phase(0)[0])
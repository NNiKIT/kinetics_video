import torch
import torch.utils.data as data  
from PIL import Image

import os 
import math 
import functools
import json
import copy
import random



class Consecutive(data.Dataset):
    def __init__(self, dataset='kinetics700-2020', split=1, train=True, 
                 sample_frames=64, interval=1, transform=None, test_mode='i3d'):
        '''
        Kinetics: root = './videos_dataset/Kinetics/kinetics_700-2020_jpgs'      
        '''

        # read data list file
        
        dataset == 'kinetics700-2020'
        self.root = './videos_dataset/Kinetics/kinetics_700-2020_jpgs' 
        datalist_file = 'train.json' if train else 'validate.json'
        
        
        datalist_path = os.path.join(self.root, datalist_file)

        with open(datalist_path, 'r') as f:
            self.data_list = json.load(f)

        self.train = train
        self.dataset = dataset
        self.sample_frames = sample_frames
        self.interval = interval
        self.transform = transform
        self.test_mode = test_mode
        self.num_classes = {'kinetics700-2020':400}[dataset]

    def __getitem__(self, index):
        data = self.data_list[index]

        path = data['path']
        label = data['label']
        num_frames = data['num_frames']
        class_name = data['class_name']

        if self.train:
            frame_indices = self._get_indices(data)
            assert len(frame_indices) == self.sample_frames//self.interval
        elif self.test_mode == 'Res3D':
            clips_num = 10
            frame_indices = self._get_clips_indices(data, clips_num)
        
        
        
        # print(len(frame_indices))
        # print(frame_indices)
        # return len(frame_indices)

        video = self._frames_loader(path, frame_indices) 
        # # T, C, H, W
        # # print(len(video))

        if self.transform is not None:
            video = self.transform(video)
        
        return video, label

    def __len__(self):
        return len(self.data_list)

    def _get_indices(self, data_info):
        num_frames = data_info['num_frames']
        if num_frames <= self.sample_frames:
            indices = list(range(1, num_frames+1, 1))
            while len(indices) < self.sample_frames:
                indices.extend(range(1, num_frames+1, 1))
                # print('looping video frames')

        else:
            offset = random.choice(range(1, num_frames-self.sample_frames+1, 1))
            indices = list(range(offset, offset+self.sample_frames, 1))

        return indices[:self.sample_frames:self.interval]

    def _get_whole_indices(self, data_info):
        num_frames = data_info['num_frames']
        if num_frames <= self.sample_frames:
            indices = list(range(1, num_frames+1, 1))
            while len(indices) <= self.sample_frames:
                indices.extend(range(1, num_frames+1, 1))

            return indices[:self.sample_frames:self.interval]

        else:
            indices = list(range(1, num_frames+1, self.interval))

            return indices
    
    def _get_clips_indices(self, data_info, clips=10):
        num_frames = data_info['num_frames']
        indices = []

        clip_interval = 1
        if num_frames <= self.sample_frames + clip_interval*clips:
            
            extand_indices = list(range(1, num_frames+1, 1))
            while len(extand_indices) <= self.sample_frames + clip_interval*clips:
                extand_indices.extend(range(1, num_frames+1, 1))

            for i in range(clips):
                indices.extend(extand_indices[i : i+self.sample_frames : self.interval])
        else:
            clip_interval = int((num_frames - self.sample_frames) // clips)
            for i in range(clips):
                indices.extend(range(1+i*clip_interval, 1+i*clip_interval+self.sample_frames, self.interval))
                
        return indices
            

    def _frames_loader(self, video_dir_path, frame_indices):
        video = []
        for i in frame_indices:
            if self.dataset == 'kinetics700-2020':
                image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                        video.append(img)
            else:
                print('unexist image path', image_path)
                assert False, 'something error in frames path'

        return video

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.abspath('.'))
    # print(sys.path)

    train_set = Consecutive(dataset='kinetics700-2020', interval=2, train=False, test_mode='non_local')
    print(len(train_set))
    for i in range(len(train_set)):
        # print(i)
        if train_set[i] != 320 :
            print('error for length =', train_set[i])
            assert False, 'error'


   
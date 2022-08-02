import torch
import subprocess
import os

class LineSplitedTextDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_file_path,  split_token=" ", num_split_line=1000, filter=torch.nn.Identity(), chache_dir="dataset_chache"):
        super().__init__()
        
        self.filter = filter
        self.split_token = split_token
        self.num_split_line = num_split_line
        self.chache_dir_path = chache_dir
        # initialize chache
        if not os.path.exists(self.chache_dir_path):
            os.mkdir(self.chache_dir_path)

        # load lines
        lines = []
        for path in list_of_file_path:
            with open(path) as f:
                lines += f.read().split("\n")

        # save to chache
        c = 0
        for i in range(0, len(lines)-1, num_split_line):
            l = lines[i:i+num_split_line]
            with open(os.path.join(self.chache_dir_path, f"{c}.txt"), mode="w") as f:
                f.write("\n".join(l))
            c += 1

    def __getitem__(self, idx):
        fid = idx // self.num_split_line
        with open(os.path.join(self.chache_dir_path, f"{fid}.txt")) as f:
            return self.filter(f.readline().split("\n")[idx % self.num_split_line])

    def __len__(self):
        return self.len

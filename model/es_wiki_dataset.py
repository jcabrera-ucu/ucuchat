import os
from torch.utils.data import Dataset


class CustomTxtDataset(Dataset):
    def __init__(self, files_dir):
        self.directory = files_dir
        self.file_list = os.listdir(files_dir)
        self.data = []

        for file in self.file_list:
            if file.endswith('.txt'):
                file_path = os.path.join(files_dir, file)
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

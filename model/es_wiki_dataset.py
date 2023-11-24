import os
from torch.utils.data import Dataset


class CustomTxtDataset(Dataset):
    def __init__(self, files_dir, files_list, first_article_index, last_article_index):
        self.directory = files_dir
        self.file_list = files_list
        self.data = []

        for file_index in range(first_article_index, last_article_index):
            file = self.file_list[file_index]
            if file.endswith('.txt'):
                file_path = os.path.join(files_dir, file)
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    self.data.append(text)

            else:
                # If a file is not a txt and not is the last one in the folder, we add the content of the last file
                # if the folder, the file at last_index (this because the last_article_index is calculated by int
                # division).
                if file_index < len(self.file_list):
                    file = self.file_list[last_article_index]
                    file_path = os.path.join(files_dir, file)
                    with open(file_path, 'r', encoding='utf8') as txt_file:
                        text = txt_file.read()
                        self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

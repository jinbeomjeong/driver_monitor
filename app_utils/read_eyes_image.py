import os


def read_dataset(path):
    # path = os.getcwd().split(os.sep)
    # path = path[0:len(path)-1]
    # path = os.path.join(os.sep, *path, 'dataset', 'mrl_eyes')

    dir_names = os.listdir(path)
    dir_names.sort()
    img_path_list = []

    for dir_name in dir_names:
        if os.path.isdir(os.path.join(path, dir_name)):
            file_names = os.listdir(os.path.join(path, dir_name))
            file_names.sort()

            for file_name in file_names:
                img_path_list.append([os.path.join(path, dir_name, file_name), file_name.split('_')[4]])

    return img_path_list



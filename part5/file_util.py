import os

def file_generator(root, ext=""):
    for x in os.listdir(root):
        if os.path.isfile(root + "/" + x) and x.endswith(ext):
            yield root + "/" + x



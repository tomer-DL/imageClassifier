import os
import cv2 as cv

def find_images(dir, ext):
    print(ext[0])
    if ext[0] != ".":
        ext = "." + ext
    for file in os.listdir(dir):
        if file.endswith(ext):
            resizeImg(dir + "\\" + file, "small-images\\" + file, 0.2)

def resizeImg(src, dst, factor):
    img = cv.imread(src)
    res = cv.resize(img, None, (0,0), fx=factor, fy=factor, interpolation=cv.INTER_AREA)
    cv.imwrite(dst, res)


def main():
    find_images("D:\\dst2\\IMAGE\\2018\\1", "jpg")


if __name__ == "__main__":
    main()

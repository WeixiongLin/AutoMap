"""
这个程序是为了测试将 image 转化为 k-space 的函数
"""

import numpy as np
import PIL.Image as image
import nibabel as nib
import matplotlib.pyplot as plt


def K_space(path, x, y, z):
    savepath = "k_"

    image_data = nib.load(path).get_data()
    img_2d = np.memmap  # img_2d是图像的切片
    if x >= 0:
        img_2d = image_data[x, :, :]
        savepath += "x_" + str(x) + ".jpg"
    elif y >= 0:
        img_2d = image_data[:, y, :]
        savepath += "y_" + str(y) + ".jpg"
    elif z >= 0:
        img_2d = image_data[:, :, z]
        savepath += "z_" + str(z) + ".jpg"
    
    img_k = np.fft.fft2(img_2d, norm="ortho")

    xlen = len(img_k)
    ylen = len(img_k[0])
    img_magitude = []
    for i in range(xlen):
        tmp = []
        for j in range(ylen):
            tmp.append(img_k[i, j].__abs__())
        img_magitude.append(tmp)
    
    minn = 100
    maxx = -100
    for i in range(xlen):
        for j in range(ylen):
            if img_magitude[i][j] < minn:
                minn = img_magitude[i][j]
            if img_magitude[i][j] > maxx:
                maxx = img_magitude[i][j]
    # print(minn, maxx)

    # plt.matshow(img_magitude, cmap = plt.cm.cool, vmin=0, vmax=8512)
    plt.matshow(img_magitude, vmin=0, vmax=5)

    plt.colorbar()
    plt.savefig(savepath)
    print(savepath)
    # plt.show()
    return img_k


path = "anat/T2/T2.nii"

x = -1
y = 128
z = -1
K_space(path, x, y, z)


# print(type(K_space(path, -1, -1, z)))
# print(K_space(path, -1, -1, 100))

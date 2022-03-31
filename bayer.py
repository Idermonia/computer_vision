import numpy as np

def compute_psnr(img_pred, img_gt):
    axes = np.shape(img_pred)
    h = axes[0]
    w = axes[1]
    if len(axes) == 2:
        c = 1
    else:
        c = axes[2]
    img_pred = np.array(img_pred, np.float64)
    img_gt = np.array(img_gt, np.float64)
    mse = 0
    for i in range (h):
        for j in range(w):
            for k in range (c):
                mse += (img_pred[i, j, k] - img_gt[i, j, k]) ** 2
    if mse == 0:
        raise ValueError
    return 10 * (np.log10(c*h*w*np.max(img_gt)**2) - np.log10(mse))


def get_bayer_masks(n_rows, n_cols):
    mask = np.zeros((n_rows, n_cols, 3), 'bool')
    f = np.tile([1, 0], n_cols // 2)
    s = np.tile([0, 1], n_cols // 2)
    if n_cols % 2:
        f = np.append(f, 1)
        s = np.append(s, 0)
    z = [0 for i in range(n_cols)]
    mask[::2, :, 0] = s
    mask[1::2, :, 0] = z
    mask[::2, :, 1] = f
    mask[1::2, :, 1] = s
    mask[::2, :, 2] = z
    mask[1::2, :, 2] = f
    return mask

def get_colored_img(raw_img):
    mask = get_bayer_masks(*raw_img.shape[0:2])
    mask_ret = np.zeros((raw_img.shape[0], raw_img.shape[1], 3), dtype='uint8')
    mask_ret[:, :, 0] = mask[:, :, 0] * raw_img
    mask_ret[:, :, 1] = mask[:, :, 1] * raw_img
    mask_ret[:, :, 2] = mask[:, :, 2] * raw_img
    return mask_ret

def bilinear_interpolation(colored_img):
    hight, length = colored_img.shape[0:2]
    mask = get_bayer_masks(hight, length)
    for i in range (1, hight - 1):
        for j in range (1, length - 1):
            for clr in range (3):
                if (mask[i, j, clr] == 0):
                    cnt = 0
                    sum = 0
                    colored_img[i, j, clr] = 0
                    for k in range (i - 1, i + 2):
                        for l in range (j - 1, j + 2):
                            if mask[k, l, clr]:
                                cnt += 1
                                sum += colored_img[k, l, clr]
                    colored_img[i, j, clr] = sum // cnt
    return colored_img

def  improved_interpolation(raw_img):
    hight, length = raw_img.shape[0:2]
    mask = get_bayer_masks(hight, length)
    raw_img = np.array(get_colored_img(raw_img), dtype=np.int64)
    for i in range (2, hight - 2):
        for j in range(2, length - 2):
            for clr in range (3):
                if mask[i, j, clr] == 0:
                    if clr == 1:
                        print(i, j)
                        t = (i % 2) * 2
                        raw_img[i, j, clr] = 4*raw_img[i, j, t] + 2*(raw_img[i-1, j, 1] + raw_img[i+1, j, 1] + raw_img[i, j-1, 1] +\
                            raw_img[i, j+1, 1]) - (raw_img[i-2, j, t] + raw_img[i+2, j, t] + raw_img[i, j-2, t] + raw_img[i, j+2, t])
                        raw_img[i, j, clr] //= 8
                        continue
                    elif (i + j) % 2 == 0:
                        if mask[i-1, j, clr]:
                            add1 = 4*(raw_img[i-1, j, clr] + raw_img[i+1, j, clr])
                            add2 = (raw_img[i, j-2, 1] + raw_img[i, j+2, 1])//2 - (raw_img[i-2, j, 1] + raw_img[i+2, j, 1])
                        else:
                            add1 = 4*(raw_img[i, j-1, clr] + raw_img[i, j+1, clr])
                            add2 = -(raw_img[i, j-2, 1] + raw_img[i, j+2, 1]) + (raw_img[i-2, j, 1] + raw_img[i+2, j, 1])//2

                        raw_img[i, j, clr] = add1 + 5*raw_img[i, j, 1] - (raw_img[i-1, j-1, 1] + raw_img[i-1, j+1, 1] + raw_img[i+1, j-1, 1] +\
                            raw_img[i+1, j+1, 1]) + add2
                        raw_img[i, j, clr] //= 8
                        continue
                    else:
                        t = (2 * clr + 2) % 3
                        raw_img[i, j, clr] = 6*raw_img[i, j, t] + 2*(raw_img[i-1, j-1, clr] + raw_img[i-1, j+1, clr] + raw_img[i+1, j-1, clr] +\
                            raw_img[i+1, j+1, clr]) - 3*(raw_img[i-2, j, t] + raw_img[i+2, j, t] + raw_img[i, j-2, t] + raw_img[i, j+2, t]) // 2
                        raw_img[i, j, clr] //= 8
    return np.clip(raw_img, 0, 255)

import numpy as np
from skimage.io import imread


def seam_carve(raw_img, mode, mask=None):
    sz = np.shape(raw_img[:, :, 0])
    h = sz[0]
    l = sz[1]

    bright = 0.299 * raw_img[:, :, 0] + 0.587 * raw_img[:, :, 1] + 0.114 * raw_img[:, :, 2]

    ret_m = 1
    if mask is None:
        mask = np.zeros((h, l), np.int8)
        ret_m = 0

    if mode == 'vertical shrink' or mode == 'vertical expand':
        mask = np.transpose(mask)
        bright = np.transpose(bright)
        h, l = l, h

    energy = np.zeros((h, l, 2), np.float64)  # 0 - горизонтально, 1 - вертикально складывать

    for j in range(l):  # обработка крайних столбцов
        energy[0, j, 0] = bright[1, j] - bright[0, j]
        energy[h - 1, j, 0] = bright[h - 1, j] - bright[h - 2, j]

    for i in range(h):  # обработка крайних строк
        energy[i, 0, 1] = bright[i, 1] - bright[i, 0]
        energy[i, l - 1, 1] = bright[i, l - 1] - bright[i, l - 2]

    for i in range(1, h - 1):  # обработка основного массива
        for j in range(0, l):
            energy[i, j, 0] = bright[i + 1, j] - bright[i - 1, j]

    for i in range(0, h):  # обработка основного массива
        for j in range(1, l - 1):
            energy[i, j, 1] = bright[i, j + 1] - bright[i, j - 1]

    norm = np.sqrt(np.square(energy[:, :, 0]) + np.square(energy[:, :, 1]))

    norm += mask.astype('float64') * 256 * h * l

    process = np.zeros((h, l), np.float64)
    process[0, :] = norm[0, :]
    for i in range(1, h):
        for j in range(l):
            m = process[i - 1, j]
            if j > 0:
                m = min(m, process[i - 1, j - 1])
            if j < l - 1:
                m = min(m, process[i - 1, j + 1])
            process[i, j] = m + norm[i, j]

    seam = np.zeros((h, l), np.int8)
    pos = np.argmin(process[h - 1, :])
    for i in range(h - 1, 0, -1):
        seam[i, pos] = 1
        if pos == 0:
            pos += np.argmin(process[i - 1, pos: pos + 2])
            continue
        if pos == l - 1:
            pos += np.argmin(process[i - 1, pos - 1:pos + 1]) - 1
            continue
        if pos > 0 and pos < l - 1:
            pos += np.argmin(process[i - 1, pos - 1:pos + 2]) - 1

    seam[0, pos] = 1

    if mode == 'vertical shrink' or mode == 'vertical expand':
        seam = np.transpose(seam)
        mask = np.transpose(mask)

    if ret_m:
        return raw_img, mask, seam
    else:
        return raw_img, None, seam
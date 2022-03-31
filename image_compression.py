import numpy as np
from scipy.ndimage.filters import gaussian_filter


def pca_compression(matrix, p):
    M = np.mean(matrix, axis = 1)[:,None]
    img = matrix - M
    c = np.cov(img)
    eig_val, eig_vec = np.linalg.eigh(c)
    order = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, order]
    eig_vec = eig_vec[:, :p:]
    return eig_vec, np.matmul(np.transpose(eig_vec), img), M[:, 0]


def pca_decompression(compressed):
    ans_0 = np.matmul(compressed[0][0], compressed[0][1]) + compressed[0][2][:, None]
    ans_1 = np.matmul(compressed[1][0], compressed[1][1]) + compressed[1][2][:, None]
    ans_2 = np.matmul(compressed[2][0], compressed[2][1]) + compressed[2][2][:, None]
    ans = np.zeros((np.shape(ans_0)[0],  np.shape(ans_0)[1], 3), np.float64)
    ans[:, :, 0] = ans_0
    ans[:, :, 1] = ans_1
    ans[:, :, 2] = ans_2
    return np.array(np.clip(ans, 0, 255), np.uint8)


def rgb2ycbcr(img):
    img = img.astype(np.float64)
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cb = 128 - 0.1687 * img[:, :, 0] - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 2]
    cr = 128 + 0.5 * img[:, :, 0] - 0.4187 * img[:, :, 1] - 0.0813 * img[:, :, 2]
    return np.array(np.dstack([y, cb, cr]), np.float64)


def ycbcr2rgb(img):
    img = img.astype(np.float64)
    r = img[:, :, 0] + 1.402 * (img[:, :, 2] - 128)
    g = img[:, :, 0] - 0.34414 * (img[:, :, 1] - 128) - 0.71414 * (img[:, :, 2] - 128)
    b = img[:, :, 0] + 1.77 * (img[:, :, 1] - 128)
    return np.array(np.clip(np.dstack([r, g, b]), 0, 255), np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    img_ycbcr = rgb2ycbcr(rgb_img)
    img_ycbcr[:, :, 1] = gaussian_filter(img_ycbcr[:, :, 1], 10)
    img_ycbcr[:, :, 2] = gaussian_filter(img_ycbcr[:, :, 2], 10)
    ans = ycbcr2rgb(img_ycbcr)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    img_ycbcr = rgb2ycbcr(rgb_img)
    img_ycbcr[:, :, 0] = gaussian_filter(img_ycbcr[:, :, 0], 10)
    ans = ycbcr2rgb(img_ycbcr)
    plt.imshow(ans)
    plt.savefig("gauss_2.png")


def downsampling(component):
    component = gaussian_filter(component, 10)
    return component[::2, ::2]

def dct(block):
    ans = np.zeros_like(block, np.float64)
    for u in range(8):
        for v in range(8):
            sm = 0
            a1 = a2 = 1
            if u == 0: a1 = np.sqrt(1/2)
            if v == 0: a2 = np.sqrt(1/2)
            for x in range(8):
                for y in range(8):
                    sm += block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            ans[u, v] = a1 * a2 * sm / 4
    return ans


def quantization(block, quantization_matrix):
    return np.round(block / quantization_matrix)

y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def own_quantization_matrix(default_quantization_matrix, q):
    s = 5000/ q
    if q == 100: s = 1
    if 50 <= q < 100: s = 200 -  2*q
    q = np.trunc((default_quantization_matrix * s + 50) / 100)
    q[q == 0] = 1
    return q


def zigzag(block):
    ans = []
    flag = 0
    i = j = 0
    for s in range (8):
        if flag:
            i = 0
            j = s
            while i <= s:
                ans.append(block[i, j])
                i += 1
                j -= 1
        else:
            i = s
            j = 0
            while j <= s:
                ans.append(block[i, j])
                i -= 1
                j += 1
        flag = (flag + 1) % 2
    flag = 1
    for s in range (1, 8):
        if flag:
            i = 7
            j = s
            while j <= 7:
                ans.append(block[i, j])
                j += 1
                i -= 1
        else:
            i = s
            j = 7
            while i <= 7:
                ans.append(block[i, j])
                i += 1
                j -= 1
        flag = (flag + 1) % 2
    return ans


def compression(zigzag_list):
    ans = []
    cnt = 0
    for i in zigzag_list:
        if i:
            if cnt:
                ans.append(0)
                ans.append(cnt)
                cnt = 0
            ans.append(i)
        else:
            cnt += 1
    if cnt:
        ans.append(0)
        ans.append(cnt)
    return ans


def jpeg_compression(img, quantization_matrixes):
    img_ycbcr = rgb2ycbcr(img)
    img_y = img_ycbcr[:, :, 0]
    img_cb = downsampling(img_ycbcr[:, :, 1])
    img_cr = downsampling(img_ycbcr[:, :, 2])
    img_y -= 128
    img_cb -= 128
    img_cr -= 128
    ans = [[], [], []]
    for i in range(np.shape(img_y)[0] // 8):
        for j in range(np.shape(img_y)[1] // 8):
            ans[0].append(compression(
                zigzag(quantization(dct(img_y[i * 8: i * 8 + 8, j * 8: j * 8 + 8]), quantization_matrixes[0]))))

    for i in range(np.shape(img_cb)[0] // 8):
        for j in range(np.shape(img_cb)[1] // 8):
            ans[1].append(compression(
                zigzag(quantization(dct(img_cb[i * 8: i * 8 + 8, j * 8: j * 8 + 8]), quantization_matrixes[1]))))

    for i in range(np.shape(img_cr)[0] // 8):
        for j in range(np.shape(img_cr)[1] // 8):
            ans[2].append(compression(
                zigzag(quantization(dct(img_cr[i * 8: i * 8 + 8, j * 8: j * 8 + 8]), quantization_matrixes[1]))))

    return ans


def inverse_compression(compressed_list):
    ans = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i]:
            ans.append(compressed_list[i])
            i += 1
        else:
            for j in range (compressed_list[i + 1]):
                ans.append(0)
            i += 2
    return ans


def inverse_zigzag(input):
    block = np.zeros((8, 8), np.float64)
    flag = 0
    i = j = 0
    pos = 0
    for s in range (8):
        if flag:
            i = 0
            j = s
            while i <= s:
                block[i, j] = input[pos]
                pos += 1
                i += 1
                j -= 1
        else:
            i = s
            j = 0
            while j <= s:
                block[i, j] = input[pos]
                pos += 1
                i -= 1
                j += 1
        flag = (flag + 1) % 2
    flag = 1
    for s in range (1, 8):
        if flag:
            i = 7
            j = s
            while j <= 7:
                block[i, j] = input[pos]
                pos += 1
                j += 1
                i -= 1
        else:
            i = s
            j = 7
            while i <= 7:
                block[i, j] = input[pos]
                pos += 1
                i += 1
                j -= 1
        flag = (flag + 1) % 2
    return block


def inverse_quantization(block, quantization_matrix):
    return block * quantization_matrix


def inverse_dct(block):
    ans = np.zeros_like(block, np.float)
    for x in range (8):
        for y in range (8):
            sm = 0
            for u in range (8):
                for v in range (8):
                    a_u = a_v = 1
                    if u == 0: a_u = np.sqrt(1/2)
                    if v == 0: a_v = np.sqrt(1/2)
                    sm += a_u * a_v * block[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            ans[x, y] = sm / 4
    return np.round(ans)


def upsampling(component):
    h, l = np.shape(component)
    col_ext = np.zeros((h, l * 2), np.float64)
    col_ext[:, ::2] = component
    col_ext[:, 1::2] = component
    ful_ext = np.zeros((h * 2, l * 2), np.float64)
    ful_ext[::2, :] = col_ext
    ful_ext[1::2, :] = col_ext
    return ful_ext


def jpeg_decompression(result, result_shape, quantization_matrixes):
    ans = np.zeros(result_shape, np.float64)
    ans_deg = np.zeros((result_shape[0] // 2, result_shape[1] // 2, 2), np.float64)
    i = 0
    j = 0
    blocks_in_row = result_shape[1] // 16
    for block in result[0]:
        ans[i * 8: i * 8 + 8, j * 8: j * 8 + 8, 0] = inverse_dct(
            inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[0]))
        j += 1
        if j >= blocks_in_row * 2:
            j = 0
            i += 1
    for clr in range(1, 3):
        i = 0
        j = 0
        for block in result[clr]:
            ans_deg[i * 8: i * 8 + 8, j * 8: j * 8 + 8, clr - 1] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[1]))
            j += 1
            if j >= blocks_in_row:
                j = 0
                i += 1
    ans[..., 1] = upsampling(ans_deg[..., 0])
    ans[..., 2] = upsampling(ans_deg[..., 1])
    ans += 128
    return ycbcr2rgb(ans)


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        custom = (own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p))

        axes[i // 3, i % 3].imshow(jpeg_decompression(jpeg_compression(img, custom), np.shape(img), custom))
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")
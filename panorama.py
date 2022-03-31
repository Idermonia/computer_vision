import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv


def find_orb(img, n_keypoints=1000):
    img = rgb2gray(img)
    extract = ORB(n_keypoints=n_keypoints)
    extract.detect_and_extract(img)
    return extract.keypoints, extract.descriptors


def center_and_normalize_points(points):
    points = np.array(points, np.float64)
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    c_x = np.mean(points[:, 0])
    c_y = np.mean(points[:, 1])
    n = np.mean([np.sqrt((points[i, 0] - c_x) ** 2 + (points[i, 1] - c_y) ** 2) for i in range (len(points))])
    n = np.sqrt(2) / n
    m = [[n, 0, - n * c_x],
         [0, n, - n * c_y],
         [0,     0,    1]]
    m = np.array(m, np.float64)
    return m, np.matmul(m, pointsh)


def find_homography(src_keypoints, dest_keypoints):
    m, src_keypoints = center_and_normalize_points(src_keypoints)
    m_, dest_keypoints = center_and_normalize_points(dest_keypoints)
    a = []
    for i in range (len(src_keypoints[0])):
        x1 = src_keypoints[0, i]
        y1 = src_keypoints[1, i]
        x2_ = dest_keypoints[0, i]
        y2_ = dest_keypoints[1, i]
        tmp_x = np.array([-x1, -y1, -1, 0, 0, 0, x2_ * x1, x2_ * y1, x2_], dtype=np.float64)
        a.append(tmp_x)
        tmp_y = np.array([0, 0, 0, -x1, -y1, -1, y2_ * x1, y2_ * y1, y2_], dtype=np.float64)
        a.append(tmp_y)
    a = np.array(a, np.float64)
    u, s, vh = np.linalg.svd(a)
    h = vh[-1]
    h = np.reshape(h, (3, 3))
    return np.linalg.inv(m_) @ h @ m


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=10000,
                     residual_threshold=1, return_matches=False):
    match = match_descriptors(src_descriptors, dest_descriptors);
    src = match[:, 0]
    dest = match[:, 1]
    ind = [i for i in range(len(src))]

    good = []
    cnt_max = 0
    for i in range(max_trials):
        sample = np.random.choice(ind, size=4, replace=False)
        H = find_homography(src_keypoints[src[sample]], dest_keypoints[dest[sample]])
        cnt = 0
        test = []
        src_matr = src_keypoints[src]
        dest_matr = dest_keypoints[dest]
        src_matr = ProjectiveTransform(H)(src_matr)
        check = np.sqrt(np.square(src_matr[:, 0] - dest_matr[:, 0]) + np.square(src_matr[:, 1] - dest_matr[:, 1]))
        cnt = np.sum(check < residual_threshold)
        if cnt >= cnt_max:
            cnt_max = cnt
            good = match[check < residual_threshold]
    good = np.array(good)
    H = find_homography(src_keypoints[good[:, 0]], dest_keypoints[good[:, 1]])
    if return_matches:
        return ProjectiveTransform(H), good
    else:
        return ProjectiveTransform(H)


def find_simple_center_warps(forward_transforms):
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2
    result = [None] * image_count
    result[center_index] = ProjectiveTransform()
    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + forward_transforms[i - 1]
    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i].inverse
    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    mn, mx = get_min_max_coords(tuple(get_corners(image_collection, center_warps)))
    N = np.array([[1, 0, -mn[1]],
         [0, 1, -mn[0]],
         [0, 0, 1]])
    n = ProjectiveTransform(N)
    return np.array(simple_center_warps) + n, np.array([mx[1] - mn[1], mx[0] - mn[0]], int)



def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    t = rotate_transform_matrix(transform).inverse
    img_mask = np.ones(img.shape, np.bool8)
    ans_mask = warp(img_mask, t, output_shape=output_shape)
    ans = warp(image, t, output_shape=output_shape)
    return ans, np.array(ans_mask, np.bool8)


def merge_pano(image_collection, final_center_warps, output_shape):
    fin_img = np.zeros(np.append(output_shape, 3))
    for i in range (len(image_collection)):
        img, mask = warp_image(image_collection[i], final_center_warps[i], tuple(output_shape))
        fin_img[mask] = img[mask]
    return fin_img



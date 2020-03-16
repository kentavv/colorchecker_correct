#!/usr/bin/env python3

import sys

import cv2
import numpy as np


def create_background_adjustment(fn):
    ref_img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    ref_img = cv2.medianBlur(ref_img, 5)
    ref_img = cv2.blur(ref_img, (63, 63))
    ref_img = cv2.GaussianBlur(ref_img, (63, 63), 0)

    avg_bgr = np.average(np.average(ref_img, axis=0), axis=0)
    avg_grey = np.average(avg_bgr)

    adj = avg_grey / ref_img

    print(avg_bgr, avg_grey)
    print(ref_img[0, 0], adj[0, 0], (ref_img * adj)[0, 0])

    cv2.imwrite('ref_img.png', ref_img)

    return adj


def extract_colorchecker_colors(fn, adj):
    cc_img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    cc_img = cc_img * adj
    cc_img = np.clip(cc_img, 0, 2 ** 12 - 1)
    cc_img = np.divide(cc_img, 2 ** 4)
    cc_img = 255. * np.power(cc_img / 255.,
                             1 / 2.2)  # monitors expect their input in sRGB color space (and promptly convert to linear color space, like cameras collect in, because they are physical items, counting and emiting photons. More photons are not going to hit the sensor in places where more photons already have. They are linear devices.)
    cc_img = np.clip(cc_img, 0, 255)
    cc_img = cc_img.astype(np.uint8)
    cv2.imwrite('gamma.png', cc_img)
    # exit(1)

    rows_start = 445
    cols_start = 283

    rows_step = 355
    cols_step = 355

    rows_off = 50
    cols_off = 50

    rows_sep = 41
    cols_sep = 41

    box_w = cols_step - cols_off * 2 - cols_sep
    box_h = rows_step - rows_off * 2 - rows_sep

    rows_n = 4
    cols_n = 6

    avg_colors = [None] * (rows_n * cols_n)

    for row in range(rows_n):
        for col in range(cols_n):
            cc_id = 24 - (row * cols_n + col)

            y = rows_start + row * rows_step + rows_off
            x = cols_start + col * cols_step + cols_off
            h = box_h
            w = box_w

            box = cc_img[y:y + h, x:x + w, :]
            box = cv2.medianBlur(box, 5)
            # box = cv2.blur(box, (63, 63))
            # box = cv2.GaussianBlur(box, (63, 63), 0)

            avg_color = np.average(np.average(box, axis=0), axis=0)

            # avg_colors[cc_id - 1] = avg_color
            avg_colors[cc_id - 1] = np.flip(avg_color, 0)  # convert BGR to RGB
            # avg_colors[cc_id - 1] = avg_color[::-1]

            cv2.imwrite('{0:02d}_{1}_{2}.png'.format(cc_id, row, col), box)

    return avg_colors


def create_eqn_terms(x):
    # return x
    # return np.append(x, [1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]**4, x[1]**4, x[2]**4, x[0]*x[1], x[0]*x[2], x[1]*x[2], 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, x[0]*x[1]*x[2], 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3/255., x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], (x[0]*x[1]/255.)**2, (x[0]*x[2]/255.)**2, (x[1]*x[2]/255.)**2, x[0]*x[1]*x[2]/255., 1])
    return np.append(x, [x[0] ** 2, x[1] ** 2, x[2] ** 2, x[0] ** 3., x[1] ** 3, x[2] ** 3, x[0] * x[1], x[0] * x[2], x[1] * x[2], (x[0] * x[1]) ** 2,
                         (x[0] * x[2]) ** 2, (x[1] * x[2]) ** 2, x[0] * x[1] * x[2], 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, x[0]*x[1]*x[2], 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, x[0]*x[1]*x[2], (x[0]*x[1]*x[2])**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, (x[0]*x[1]*x[2])**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, (x[0]*x[1]*x[2])**2, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, (x[0]*x[1])**3, (x[0]*x[2])**3, (x[1]*x[2])**3, 1])
    # return np.append(x, [x[0]**2, x[1]**2, x[2]**2, x[0]**3, x[1]**3, x[2]**3, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]*x[1]*x[2], (x[0]*x[1])**2, (x[0]*x[2])**2, (x[1]*x[2])**2, (x[0]*x[1]*x[2])**2, (x[0]*x[1])**3, (x[0]*x[2])**3, (x[1]*x[2])**3, (x[0]*x[1]*x[2])**3, 1])


def fit_colorchecker_equation(avg_colors):
    xrite_target_colors = [
        {'id': 1, 'rgb': (115, 82, 68), 'lab': (37.986, 13.555, 14.059), 'name': 'dark skin', 'munsell': '3 YR 3.7 / 3.2'},
        {'id': 2, 'rgb': (194, 150, 130), 'lab': (65.711, 18.130, 17.810), 'name': 'light skin', 'munsell': '2.2 YR 6.47 / 4.1'},
        {'id': 3, 'rgb': (98, 122, 157), 'lab': (49.927, -4.880, -21.925), 'name': 'blue sky', 'munsell': '4.3 PB 4.95 / 5.5'},
        {'id': 4, 'rgb': (87, 108, 67), 'lab': (43.139, -13.095, 21.905), 'name': 'foliage', 'munsell': '6.7 GY 4.2 / 4.1'},
        {'id': 5, 'rgb': (133, 128, 177), 'lab': (55.112, 8.844, -25.399), 'name': 'blue flower', 'munsell': '9.7 PB 5.47 / 6.7'},
        {'id': 6, 'rgb': (103, 189, 170), 'lab': (70.719, -33.397, -0.199), 'name': 'bluish green', 'munsell': '2.5 BG 7 / 6'},
        {'id': 7, 'rgb': (214, 126, 44), 'lab': (62.661, 36.067, 57.096), 'name': 'orange', 'munsell': '5 YR 6 / 11'},
        {'id': 8, 'rgb': (80, 91, 166), 'lab': (40.020, 10.410, -45.964), 'name': 'purplish blue', 'munsell': '7.5 PB 4 / 10.7'},
        {'id': 9, 'rgb': (193, 90, 99), 'lab': (51.124, 48.239, 16.248), 'name': 'moderate red', 'munsell': '2.5 R 5 / 10'},
        {'id': 10, 'rgb': (94, 60, 108), 'lab': (30.325, 22.976, -21.587), 'name': 'purple', 'munsell': '5 P 3 / 7'},
        {'id': 11, 'rgb': (157, 188, 64), 'lab': (72.532, -23.709, 57.255), 'name': 'yellow green', 'munsell': '5 GY 7.1 / 9.1'},
        {'id': 12, 'rgb': (224, 163, 46), 'lab': (71.941, 19.363, 67.857), 'name': 'orange yellow', 'munsell': ' 10 YR 7 / 10.5'},
        {'id': 13, 'rgb': (56, 61, 150), 'lab': (28.778, 14.179, -50.297), 'name': 'blue', 'munsell': '7.5 PB 2.9 / 12.7'},
        {'id': 14, 'rgb': (70, 148, 73), 'lab': (55.261, -38.342, 31.370), 'name': 'green', 'munsell': '0.25 G 5.4 / 8.65'},
        {'id': 15, 'rgb': (175, 54, 60), 'lab': (42.101, 53.378, 28.190), 'name': 'red', 'munsell': '5 R 4 / 12'},
        {'id': 16, 'rgb': (231, 199, 31), 'lab': (81.733, 4.039, 79.819), 'name': 'yellow', 'munsell': '5 Y 8 / 11.1'},
        {'id': 17, 'rgb': (187, 86, 149), 'lab': (51.935, 49.986, -14.574), 'name': 'magenta', 'munsell': '2.5 RP 5 / 12'},
        {'id': 18, 'rgb': (8, 133, 161), 'lab': (51.038, -28.631, -28.638), 'name': 'cyan', 'munsell': '5 B 5 / 8'},
        {'id': 19, 'rgb': (243, 243, 242), 'lab': (96.539, -0.425, 1.186), 'name': 'white (.05 *)', 'munsell': 'N 9.5 / '},
        {'id': 20, 'rgb': (200, 200, 200), 'lab': (81.257, -0.638, -0.335), 'name': 'neutral 8 (.23 *)', 'munsell': 'N 8 / '},
        {'id': 21, 'rgb': (160, 160, 160), 'lab': (66.766, -0.734, -0.504), 'name': 'neutral 6.5 (.44 *)', 'munsell': 'N 6.5 / '},
        {'id': 22, 'rgb': (122, 122, 121), 'lab': (50.867, -0.153, -0.270), 'name': 'neutral 5 (.70 *)', 'munsell': 'N 5 / '},
        {'id': 23, 'rgb': (85, 85, 85), 'lab': (35.656, -0.421, -1.231), 'name': 'neutral 3.5 (1.05 *)', 'munsell': 'N 3.5 / '},
        {'id': 24, 'rgb': (52, 52, 52), 'lab': (20.461, -0.079, -0.973), 'name': 'black (1.50 *)', 'munsell': 'N 2 / '}
    ]

    a = np.array([create_eqn_terms(x) for x in avg_colors])
    b = np.array([(x['rgb'][0], x['rgb'][1], x['rgb'][2]) for x in xrite_target_colors])
    # b = np.array([(x['lab'][0], x['lab'][1], x['lab'][2]) for x in xrite_target_colors])

    rst = np.linalg.lstsq(a, b, rcond=None)
    x, residuals, rank, s = rst

    # residuals will be non-empty only if the equations are over-determined
    np.set_printoptions(precision=2, suppress=False, floatmode='fixed')
    print('x:', x, sep='\n')
    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
    print('residuals', residuals)
    print('rank', rank)
    print('s', s)
    print('s (per)', s / np.sum(s))

    # print(a[:2])
    # print(b[:2])
    # print(x.shape)
    # print(a.shape)
    ##res = np.matmul(a, x)
    res = a @ x
    err = b - res
    # with np.printoptions(precision=3, suppress=True):
    if 1:
        print('Sample\tActual\tTarget\tProjected\tError')
        np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
        for i in range(res.shape[0]):
            print(i + 1, avg_colors[i], b[i], res[i], err[i], sep='\t')

    return x


def background_correct_and_apply_colorcheck_correction(fn, adj, x):
    cc_img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    cc_img = np.clip(cc_img, 0, 2 ** 12 - 1)
    cc_img = np.divide(cc_img, 2 ** 4)
    cc_img = np.clip(cc_img, 0, 255)
    cc_img = cc_img.astype(np.uint8)
    cv2.imwrite('project_raw.png', cc_img)

    cc_img = cv2.cvtColor(cc_img, cv2.COLOR_BGR2RGB)
    cc_img = cc_img[:500, :500, :].copy()
    cc_img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    cc_img = cc_img * adj
    cc_img = np.clip(cc_img, 0, 2 ** 12 - 1)
    cc_img = np.divide(cc_img, 2 ** 4)
    cc_img = 255. * np.power(cc_img / 255., 1 / 2.2)
    cc_img = np.clip(cc_img, 0, 255)
    cc_img = cc_img.astype(np.uint8)
    cv2.imwrite('project_raw_gamma.png', cc_img)

    cc_img = cv2.cvtColor(cc_img, cv2.COLOR_BGR2RGB)
    # cc_img = cc_img[:500, :500, :].copy()

    s = cc_img.shape
    cc_img.shape = [s[0] * s[1], s[2]]

    aa = cc_img.astype(float)
    aa = np.apply_along_axis(create_eqn_terms, 1, aa)

    aa = aa @ x
    aa = np.clip(aa, 0, 255).astype(np.uint8)

    aa.shape = s
    aa = cv2.cvtColor(aa, cv2.COLOR_RGB2BGR)
    return aa


def main():
    ref_fn = 'reference_20180916165921.png'
    cc_img_fn = 'colorchecker_20180916165853.png'
    test_fn = 'colorchecker_20180916165853.png'

    if len(sys.argv) == 4:
        ref_fn = sys.argv[1]
        cc_img_fn = sys.argv[2]
        test_fn = sys.argv[3]

    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

    adj = create_background_adjustment(ref_fn)
    avg_colors = extract_colorchecker_colors(cc_img_fn, adj)
    # print(avg_colors)
    x = fit_colorchecker_equation(avg_colors)
    corrected_img = background_correct_and_apply_colorcheck_correction(test_fn, adj, x)
    cv2.imwrite('project.png', corrected_img)
 
 
if __name__== "__main__":
    main()


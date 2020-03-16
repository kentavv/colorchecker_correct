#!/usr/bin/env python3

import sys

import cv2
import numpy as np


def extract_colorchecker_colors2(cc_img):
    cc_img2 = np.clip(cc_img, 0, 2 ** 14 - 1)
    cc_img2 = np.divide(cc_img2, 2 ** 6)
    cc_img2 = np.clip(cc_img2, 0, 255)
    cc_img2 = cc_img2.astype(np.uint8)
    cv2.imwrite('linear.png', cc_img2)

    cc_img2 = np.clip(cc_img, 0, 2 ** 14 - 1)
    cc_img2 = np.divide(cc_img2, 2 ** 6)
    # Monitors expect their input in sRGB color space (and promptly convert to
    # linear color space, like cameras collect in, because they are physical
    # items, counting and emitting photons. More photons are not going to hit
    # the sensor in places where more photons already have. They are linear
    # devices.)
    cc_img2 = 255. * np.power(cc_img2 / 255., 1 / 2.2)
    cc_img2 = np.clip(cc_img2, 0, 255)
    cc_img2 = cc_img2.astype(np.uint8)
    cv2.imwrite('gamma.png', cc_img2)


def extract_colorchecker_colors(cc_img):
    cc_img = np.clip(cc_img, 0, 2 ** 14 - 1)
    cc_img = np.divide(cc_img, 2 ** 6)
    # Monitors expect their input in sRGB color space (and promptly convert to
    # linear color space, like cameras collect in, because they are physical
    # items, counting and emitting photons. More photons are not going to hit
    # the sensor in places where more photons already have. They are linear
    # devices.)
    cc_img = 255. * np.power(cc_img / 255., 1 / 2.2)
    cc_img = np.clip(cc_img, 0, 255)
    cc_img = cc_img.astype(np.uint8)
    # cv2.imwrite('gamma.png', cc_img)
    # sys.exit(1)

    rows_start = 1222
    cols_start = 1534

    rows_step = 534
    cols_step = 534

    rows_off = 50
    cols_off = 50

    rows_sep = 120
    cols_sep = 120

    box_w = cols_step - cols_off * 2 - cols_sep
    box_h = rows_step - rows_off * 2 - rows_sep

    rows_n = 4
    cols_n = 6

    avg_colors = [None] * (rows_n * cols_n)

    for row in range(rows_n):
        for col in range(cols_n):
            cc_id = (row * cols_n + col)

            y = rows_start + row * rows_step + rows_off
            x = cols_start + col * cols_step + cols_off
            h = box_h
            w = box_w

            box = cc_img[y:y + h, x:x + w, :]
            box = cv2.medianBlur(box, 5)
            # box = cv2.blur(box, (63, 63))
            # box = cv2.GaussianBlur(box, (63, 63), 0)

            avg_color = np.average(np.average(box, axis=0), axis=0)

            avg_colors[cc_id] = np.flip(avg_color, 0)  # convert BGR to RGB

            cv2.imwrite('{0:02d}_{1}_{2}.png'.format(cc_id, row, col), box)

    return avg_colors


def create_eqn_terms(x):
    # The [x[0], x[1], x[2]] terms are already present, as x. Here we are appending new terms.

    # return x
    # return np.append(x, [1])
    # return np.append(x, [x[0] ** 2, x[1] ** 2, x[2] ** 2, 1])
    return np.append(x, [x[0] ** 2, x[1] ** 2, x[2] ** 2,
                         x[0] ** 3, x[1] ** 3, x[2] ** 3, 1])
    # return np.append(x, [x[0] ** 2, x[1] ** 2, x[2] ** 2,
    #                      x[0] ** 3., x[1] ** 3, x[2] ** 3,
    #                      x[0] * x[1], x[0] * x[2], x[1] * x[2],
    #                      (x[0] * x[1]) ** 2, (x[0] * x[2]) ** 2, (x[1] * x[2]) ** 2,
    #                      x[0] * x[1] * x[2], 1])


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
    err1 = avg_colors - b
    err2 = res - b
    # with np.printoptions(precision=3, suppress=True):
    if 1:
        print('Sample\tMeasured\tProjected\tTarget\tError1\tsum(abs(Error1))\tError2\tsum(abs(Error2))')
        np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
        for i in range(res.shape[0]):
            print(i + 1, avg_colors[i], res[i], b[i], err1[i], f'{np.sum(np.abs(err1[i])):.2f}', err2[i], f'{np.sum(np.abs(err2[i])):.2f}', sep='\t')
    print(f'sum(error1): {np.sum(err1):.2f}')
    print(f'sum(abs(error1)): {np.sum(np.abs(err1)):.2f}')
    print(f'sum(error2): {np.sum(err2):.2f}')
    print(f'sum(abs(error2)): {np.sum(np.abs(err2)):.2f}')

    return x


def background_correct_and_apply_colorcheck_correction(cc_img, x):
    # cc_img = np.clip(cc_img, 0, 2 ** 14 - 1)
    # cc_img = np.divide(cc_img, 2 ** 6)
    # cc_img = np.clip(cc_img, 0, 255)
    # cc_img = cc_img.astype(np.uint8)
    # cv2.imwrite('project_raw.png', cc_img)

    # cc_img = cv2.cvtColor(cc_img, cv2.COLOR_BGR2RGB)
    # cc_img = cc_img[:500, :500, :].copy()
    # cc_img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)

    cc_img = np.clip(cc_img, 0, 2 ** 14 - 1)
    cc_img = np.divide(cc_img, 2 ** 6)
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
    # A raw image from a camera contains the photon counts
    # at each photosite. The raw image is only effected by the ISO setting.
    # The white balance setting does not effect the raw image. However,
    # the camera selected white balance can be applied with dcraw_emu -w.
    # The color space used in the raw image
    # depends on the filters used at the photosites. To convert this a
    # standard color space, e.g. sRGB, a transformation must be applied. Until
    # then, red, green, blue, may be different than expected. Dcraw_emu -o 1
    # will convert the camera raw RGB to sRGB. This conversion is independent of
    # gamma application. Before a white balance is applied, the raw image's RGB
    # colors is dependent on the photosite filters. Can each camera, even of
    # the same make-model, have different filters? For dcraw_emu, if -w is set,
    # the raw embedded color profile will be used; the default -M setting is to
    # use the embedded color profile only if white balance correction (-w)
    # is used. Dcraw_emu can use a darkfield image and take a list of bad
    # pixels.
    # https://www.libraw.org/docs/API-datastruct-eng.html

    # dcraw_emu -6 -W -g 1 1 -w -o 0 IMG_5963.CR2
    cc_img_fn = 'IMG_5963.CR2.ppm'
    # dcraw_emu -6 -W -g 1 1 -w -o 0 IMG_5972.CR2
    # test_fn = 'IMG_5963.CR2.ppm'
    test_fn = 'IMG_5972.CR2.ppm'

    if len(sys.argv) == 3:
        cc_img_fn = sys.argv[1]
        test_fn = sys.argv[2]

    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

    cc_img = cv2.imread(cc_img_fn, cv2.IMREAD_UNCHANGED)
    extract_colorchecker_colors2(cc_img)
    sys.exit(1)
    avg_colors = extract_colorchecker_colors(cc_img)
    print(avg_colors)

    x = fit_colorchecker_equation(avg_colors)

    cc_img = cv2.imread(test_fn, cv2.IMREAD_UNCHANGED)
    corrected_img = background_correct_and_apply_colorcheck_correction(cc_img, x)
    cv2.imwrite('project.png', corrected_img)


if __name__ == "__main__":
    main()

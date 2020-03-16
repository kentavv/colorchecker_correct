#!/usr/bin/env python3

# Using a reference image, normally a uniform gray surface, normalize and save an image.

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


def main():
    ref_fn = 'reference_20180916165921.png'
    in_fn = 'image_20180916165921.png'
    out_fn = 'adjusted_20180916165921.png'

    if len(sys.argv) == 4:
        ref_fn = sys.argv[1]
        in_fn = sys.argv[2]
        out_fn = sys.argv[3]

    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

    adj = create_background_adjustment(ref_fn)
    img = cv2.imread(in_fn, cv2.IMREAD_UNCHANGED)
    img *= adj
    cv2.imwrite(out_fn, img)


if __name__ == "__main__":
    main()

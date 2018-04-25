import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def add_lines_to_img(img: np.ndarray, rho_arrays, vert_colour=(0, 255, 0), horiz_colour=(0, 255, 0)) -> None:
    height, width = img.shape[:2]
    for dimension, arr in enumerate(rho_arrays):
        vert = dimension == 0
        for rho_flt in arr:
            rho = int(rho_flt)
            p1 = (rho, 0) if vert else (0, rho)
            p2 = (rho, int(height)) if vert else (int(width), rho)
            cv2.line(img, p1, p2, vert_colour if vert else horiz_colour, 2)


def main():
    img = cv2.imread('tenders.png')
    # cv2 decodes channels as B G R
    orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    plt.subplot(221), plt.imshow(orig_rgb, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    height, width = img.shape[:2]
    min_line_len = min(height, width) * 0.66

    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(min_line_len))

    num_lines = len(lines)

    # vertical, horizontal
    rho_by_orientation = [[], []]

    for i in range(0, num_lines):
        for rho, theta in lines[i]:
            rho_degrees = math.floor(theta * 180 / np.pi)
            is_vertical = rho_degrees == 0
            is_horizontal = rho_degrees == 90
            # discard lines that aren't part of grid
            if not (is_vertical or is_horizontal):
                continue

            rho_by_orientation[0 if is_vertical else 1].append(rho)

    hough_img = orig_rgb.copy()
    add_lines_to_img(hough_img, rho_by_orientation, (255, 0, 0), (0, 0, 255))

    plt.subplot(223), plt.imshow(hough_img)
    plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])

    bounds = []
    merge_tolerance_pct = 5

    for dimension, arr in enumerate(rho_by_orientation):
        arr.sort()
        print('Hough {}s'.format('vertical' if dimension == 0 else 'horizontal'), arr)
        min_rho = arr[0]
        max_rho = arr[-1]
        diff_rho = max_rho - min_rho
        bounds.append((min_rho, max_rho))
        for i, rho in enumerate(arr):
            if not rho or rho == max_rho:
                continue
            acc_rho = rho
            merge_count = 1
            for j, comp_rho in enumerate(arr):
                if abs(comp_rho - rho) < (diff_rho * (merge_tolerance_pct / 100)):
                    # merge similar lines
                    acc_rho += comp_rho
                    merge_count += 1
                    arr[j] = False
            arr[i] = acc_rho / merge_count
        rho_by_orientation[dimension] = [int(r) for r in arr if r]
        print('Merged {}s'.format('vertical' if dimension == 0 else 'horizontal'), rho_by_orientation[dimension])

    print('bounds', bounds)

    merged_hough = orig_rgb.copy()
    add_lines_to_img(merged_hough, rho_by_orientation)

    plt.subplot(224), plt.imshow(merged_hough)
    plt.title('Hough Transform with Merge'), plt.xticks([]), plt.yticks([])

    def quit_figure(event):
        if event.key:
            plt.close(event.canvas.figure)

    plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

    plt.show()


if __name__ == "__main__":
    main()

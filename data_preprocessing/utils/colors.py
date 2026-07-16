import numpy as np
import math
import colorsys


COLORS_MAP = {'bird': {'initial': np.array([45, 169, 145]) / 255., 'eye': np.array([243, 156, 18]) / 255.,
                          'none': np.array([149, 165, 166]) / 255.,
                          'beak': np.array([211, 84, 0]) / 255., 'body': np.array([41, 128, 185]) / 255.,
                          'details': np.array([171, 190, 191]) / 255.,
                          'head': np.array([192, 57, 43]) / 255., 'legs': np.array([142, 68, 173]) / 255.,
                          'mouth': np.array([39, 174, 96]) / 255.,
                          'tail': np.array([69, 85, 101]) / 255., 'wings': np.array([127, 140, 141]) / 255.},
              'creature': {'initial': np.array([45, 169, 145]) / 255., 'eye': np.array([243, 156, 18]) / 255.,
                              'none': np.array([149, 165, 166]) / 255.,
                              'arms': np.array([211, 84, 0]) / 255., 'beak': np.array([41, 128, 185]) / 255.,
                              'mouth': np.array([54, 153, 219]) / 255.,
                              'body': np.array([192, 57, 43]) / 255., 'ears': np.array([142, 68, 173]) / 255.,
                              'feet': np.array([39, 174, 96]) / 255.,
                              'fin': np.array([69, 85, 101]) / 255., 'hair': np.array([127, 140, 141]) / 255.,
                              'hands': np.array([45, 63, 81]) / 255.,
                              'head': np.array([241, 197, 17]) / 255., 'horns': np.array([51, 205, 117]) / 255.,
                              'legs': np.array([232, 135, 50]) / 255.,
                              'nose': np.array([233, 90, 75]) / 255., 'paws': np.array([160, 98, 186]) / 255.,
                              'tail': np.array([58, 78, 99]) / 255.,
                              'wings': np.array([198, 203, 207]) / 255., 'details': np.array([171, 190, 191]) / 255.}
              }


def generate_colors2(N, fix_color=False, divide=11, order=[0, 4, 8, 1, 6, 10, 2, 5, 9, 3, 7], replace_interval=[0.12, 0.3], shift=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if fix_color:
        fixed_colors = \
            [(1.0, 0.0, 0.9473684210526319), (0.4210526315789469, 0.0, 1.0), (0.0, 1.0, 0.8421052631578947),
             (1.0, 0.0, 0.0), (0.10526315789473717, 0.0, 1.0), (1.0, 0.9473684210526315, 0.0),
             (0.42105263157894735, 1.0, 0.0), (1.0, 0.0, 0.6315789473684212), (0.0, 0.5263157894736841, 1.0),
             (1.0, 0.631578947368421, 0.0), (0.0, 0.2105263157894739, 1.0), (0.10526315789473695, 1.0, 0.0),
             (1.0, 0.0, 0.3157894736842106), (1.0, 0.3157894736842105, 0.0), (0.0, 1.0, 0.5263157894736841),
             (0.7368421052631575, 0.0, 1.0), (0.736842105263158, 1.0, 0.0), (0.0, 0.8421052631578947, 1.0),
             (0.0, 1.0, 0.21052631578947345),
             (0.33333333333333326, 1.0, 0.0), (1.0, 0.0, 0.3333333333333339), (1.0, 0.33333333333333326, 0.0),
             (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
             (0.33333333333333304, 0.0, 1.0), (0.0, 1.0, 0.3333333333333335), (0.0, 0.0, 1.0),
             (0.666666666666667, 0.0, 1.0), (1.0, 0.0, 0.666666666666667), (0.0, 0.33333333333333304, 1.0),
             (1.0, 0.6666666666666666, 0.0), (0.6666666666666667, 1.0, 0.0), (0.0, 1.0, 0.6666666666666665),
             (0.0, 0.6666666666666665, 1.0),
             ]
        return fixed_colors
    else:
        N_new = int(math.ceil(N / float(divide)) * divide)

        brightness = 1.0
        hsv = [(i / N_new, 1, brightness) for i in range(N_new)]
        colors_bright = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        brightness = 0.9
        hsv = [(i / N_new, 1, brightness) for i in range(N_new)]
        colors_dark = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        colors = np.array(colors_bright)
        colors_dark = np.array(colors_dark)
        replace_start = int(replace_interval[0] * N)
        replace_end = int(replace_interval[1] * N)
        colors[replace_start:replace_end] = colors_dark[replace_start:replace_end]

        # ori_img = np.zeros((20, 20 * N, 3))
        # for i in range(N):
        #     ori_img[:, i*20: (i+1)*20] = colors[i] * 255.0
        # ori_img = Image.fromarray(ori_img.astype(np.uint8), 'RGB')
        # ori_img.save('testData/ori_color.png')

        colors = colors.reshape((divide, -1, 3))
        assert len(order) == divide
        sort_index = np.argsort(order)
        colors_new = [colors[i] for i in sort_index]
        colors = np.stack(colors_new, axis=0)
        colors = np.transpose(colors, (1, 0, 2)).reshape((-1, 3))
        if shift > 0:
            colors = np.concatenate([colors[shift:], colors[0:shift]], axis=0)

        # new_img = np.zeros((20, 20 * N, 3))
        # for i in range(N):
        #     new_img[:, i*20: (i+1)*20] = colors[i] * 255.0
        # new_img = Image.fromarray(new_img.astype(np.uint8), 'RGB')
        # new_img.save('testData/new_color.png')

        return colors
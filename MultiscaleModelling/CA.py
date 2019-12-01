import numpy as np
import random
import cv2
from tkinter import *
import tkinter as tk
from PIL import Image
import csv

import time

KERNEL = {0: ("VON_NEUMAN", [(-1, 0), (1, 0), (0, -1), (0, 1)]),
          1: ("MOORE", [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]),
          2: ("PENTA_RIGHT", [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)]),
          3: ("PENTA_LEFT", [(-1, -1), (-1, 0), (0, -1), (1, -1), (1, 0)]),
          4: ("HEXA_RIGHT", [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]),
          5: ("HEXA_LEFT", [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)])
          }

COLORS = {0: [255, 255, 255],
          1: [64, 10, 4],
          2: [164, 217, 7],
          3: [230, 142, 234],
          4: [111, 132, 208],
          5: [176, 149, 251],
          6: [111, 255, 254],
          7: [169, 233, 171],
          8: [194, 91, 110],
          9: [15, 140, 196],
          10: [96, 154, 41],
          11: [90, 113, 134],
          12: [126, 179, 10],
          13: [204, 89, 26],
          14: [129, 67, 147],
          15: [229, 40, 175],
          16: [96, 131, 16],
          17: [125, 133, 87],
          18: [164, 51, 152],
          19: [178, 218, 236],
          20: [11, 107, 134],
          21: [74, 145, 151],
          22: [186, 112, 85],
          23: [241, 76, 149],
          24: [250, 184, 111],
          25: [224, 61, 0],
          26: [207, 121, 104],
          27: [149, 22, 24],
          28: [75, 16, 73],
          29: [158, 187, 252],
          30: [135, 234, 169],
          31: [210, 53, 216],
          32: [203, 224, 90],
          33: [214, 241, 202],
          34: [162, 211, 118],
          35: [200, 65, 207],
          36: [4, 54, 251],
          37: [139, 125, 25],
          38: [100, 76, 169],
          39: [42, 194, 130],
          40: [157, 134, 84],
          41: [109, 63, 159],
          42: [185, 115, 59],
          43: [250, 88, 253],
          44: [158, 55, 89],
          45: [121, 207, 41],
          46: [223, 173, 113],
          47: [197, 73, 57],
          48: [246, 113, 157],
          49: [79, 13, 168],
          50: [80, 12, 23],
          51: [228, 151, 202],
          52: [49, 59, 187],
          53: [23, 207, 166],
          54: [20, 124, 89],
          55: [138, 38, 217],
          56: [30, 103, 108],
          57: [9, 149, 185],
          58: [253, 236, 84],
          59: [45, 26, 38],
          60: [153, 116, 32],
          61: [18, 96, 174],
          62: [89, 60, 37],
          63: [171, 220, 150],
          64: [9, 44, 130],
          65: [211, 49, 58],
          66: [100, 60, 102],
          67: [111, 39, 209],
          68: [8, 194, 80],
          69: [207, 155, 193],
          70: [238, 175, 97],
          71: [101, 230, 148],
          72: [49, 92, 37],
          73: [227, 2, 179],
          74: [147, 234, 233],
          75: [167, 89, 85],
          76: [245, 215, 55],
          77: [160, 165, 241],
          78: [162, 99, 150],
          79: [138, 194, 47],
          80: [116, 182, 115],
          81: [27, 1, 83],
          82: [196, 88, 18],
          83: [8, 112, 95],
          84: [223, 37, 150],
          85: [124, 116, 116],
          86: [203, 51, 88],
          87: [156, 87, 147],
          88: [121, 100, 149],
          89: [10, 157, 182],
          90: [233, 20, 115],
          91: [82, 239, 11],
          92: [185, 176, 129],
          93: [175, 19, 230],
          94: [208, 165, 197],
          95: [240, 130, 77],
          96: [240, 58, 149],
          97: [209, 20, 154],
          98: [59, 244, 216],
          99: [5, 113, 65],
          100: [105, 106, 168],
          101: [82, 210, 52],
          102: [242, 210, 237],
          103: [248, 5, 10],
          104: [52, 171, 223],
          105: [96, 113, 131],
          106: [223, 176, 196],
          107: [226, 122, 20],
          108: [190, 192, 4],
          109: [165, 143, 131],
          110: [182, 70, 46],
          111: [173, 4, 218],
          112: [199, 200, 150],
          113: [3, 57, 21],
          114: [131, 193, 73],
          115: [254, 27, 199],
          116: [212, 232, 181],
          117: [200, 58, 143],
          118: [51, 57, 224],
          119: [203, 168, 62],
          120: [69, 248, 208],
          121: [40, 65, 244],
          122: [245, 227, 245],
          123: [228, 17, 252],
          124: [230, 176, 204],
          125: [106, 229, 4],
          126: [210, 202, 17],
          127: [222, 249, 179],
          128: [125, 100, 37],
          129: [92, 198, 242],
          130: [99, 96, 112],
          131: [201, 66, 1],
          132: [131, 68, 51],
          133: [36, 168, 56],
          134: [43, 139, 201],
          135: [79, 111, 249],
          136: [89, 12, 240],
          137: [139, 231, 89],
          138: [63, 141, 140],
          139: [210, 5, 46],
          140: [52, 35, 109],
          141: [19, 231, 137],
          142: [250, 137, 116],
          143: [109, 222, 146],
          144: [166, 167, 97],
          145: [208, 111, 40],
          146: [74, 64, 182],
          147: [213, 16, 248],
          148: [11, 12, 65],
          149: [75, 133, 123],
          150: [211, 90, 191],
          151: [63, 118, 173],
          152: [14, 61, 139],
          153: [22, 117, 134],
          154: [78, 55, 161],
          155: [111, 222, 230],
          156: [11, 135, 103],
          157: [1, 142, 224],
          158: [210, 168, 129],
          159: [168, 22, 195],
          160: [132, 62, 55],
          161: [2, 158, 102],
          162: [74, 160, 5],
          163: [226, 90, 230],
          164: [49, 224, 55],
          165: [10, 9, 93],
          166: [74, 33, 157],
          167: [21, 196, 52],
          168: [109, 22, 255],
          169: [189, 54, 181],
          170: [14, 159, 159],
          171: [19, 175, 252],
          172: [245, 106, 201],
          173: [47, 217, 163],
          174: [115, 19, 139],
          175: [200, 39, 169],
          176: [214, 107, 142],
          177: [55, 194, 175],
          178: [253, 169, 241],
          179: [58, 94, 160],
          180: [39, 16, 161],
          181: [193, 26, 249],
          182: [183, 51, 102],
          183: [36, 121, 243],
          184: [160, 137, 222],
          185: [184, 221, 105],
          186: [28, 138, 214],
          187: [128, 111, 202],
          188: [167, 47, 82],
          189: [176, 27, 55],
          190: [127, 162, 10],
          191: [68, 200, 8],
          192: [82, 241, 55],
          193: [109, 163, 24],
          194: [81, 215, 201],
          195: [139, 52, 19],
          196: [203, 158, 100],
          197: [116, 81, 206],
          198: [164, 80, 202],
          199: [70, 128, 145]
}


grain_no = 300
rows = 500
columns = 500
inclusion = 4
array = np.zeros([rows, columns], dtype=np.uint8)
start = 0
kernel = KERNEL[0][1]
image = np.zeros([rows, columns, 3])
main_screen = tk.Tk()
row_type = IntVar()
row_type.set(rows)
col_type = IntVar()
col_type.set(columns)
grain_type = IntVar()
grain_type.set(grain_no)
inclusion_type = IntVar()
inclusion_type.set(inclusion)


def array_init():
    global array
    global rows
    global columns
    global row_type
    global col_type
    rows = row_type.get()
    columns = col_type.get()
    tab = np.zeros((rows, columns), dtype=np.uint8)
    array = tab


def int_gen(range):
    integer = random.randint(0, range-1)
    return integer


def data_to_image():
    global image
    img = np.zeros([rows, columns, 3], dtype=np.uint8)
    for r in range(rows):
        for c in range(columns):
            if array[r][c] == 0:
                img[r][c] = COLORS[0]
            else:
                img[r][c] = COLORS[array[r][c] % 199 + 1]
    image = np.array(img)


def fill_array():
    global array
    global image
    global grain_no
    global grain_type
    array_init()
    arr = array
    grain_no = grain_type.get()
    for i in range(grain_no):
        r = None
        c = None
        z = 1000
        while z:
            r = int_gen(rows)
            c = int_gen(columns)
            if arr[r][c] == 0:
                break
            r = None
            c = None
            z -= 1
        if r is not None and c is not None:
            arr[r][c] = i + 1
        else:
            raise StopIteration("cannot find freespace")
    array = arr
    visualize()


def generate():
    global start
    global array
    global image
    global rows
    global columns
    global wrap_type
    array_new = np.zeros((rows, columns), dtype=np.uint8)
    for r in range(0, rows):
        for c in range(0, columns):

            temp = 0
            if array[r][c] != 0:
                temp = array[r][c]
            else:
                for r_n, c_n in kernel:
                    r_n += r
                    c_n += c

                    if wrap_ckbox_var.get() == 1:
                        r_n = wrap(r_n, rows)
                        c_n = wrap(c_n, columns)
                    else:
                        pass

                    if 0 <= r_n < rows and 0 <= c_n < columns:
                        temp = array[r_n][c_n]
                        if temp != 0:
                            break
                        else:
                            pass
                    else:
                        temp = 0

            array_new[r][c] = temp


    if np.array_equal(array, array_new):
        start = 0
        array = array
    else:
        array = array_new

    if start == 1:
        pass
        visualize()
    else:
        visualize()


def wrap(x, range):
    if x < 0:
        a = range - 1
    elif x >= range:
        a = 0
    else:
        a = x
    return a

def generate_continuous():
    global start
    start = not start


def temp_funct():
    # wybór jądra
    kernel = KERNEL[methods.curselection()[0]][1]

    # cv2.waitKey(0)


def visualize():
    global image
    data_to_image()
    cv2.imshow("CA", image)
    cv2.waitKey(1)

def export_to_csv():
    global array
    tab = []
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            tab.append([x, y, array[x][y]])
    tab = np.array(tab)
    np.savetxt("CA2.csv", tab, delimiter=',', fmt='%d')


def import_from_csv():
    global array
    data = np.genfromtxt('CA.csv', delimiter=',', dtype=np.uint8)
    size = data.shape[0]
    max_row = data[size-1][0] + 1
    max_column = data[size-1][1] + 1
    tabimp = np.zeros([max_row, max_column], dtype=np.uint8)
    for r in range(max_row):
        for c in range(max_column):
            # print(r*max_column + c)
            tabimp[r][c] = data[r*max_column + c][2]
    array = tabimp


def image_to_png():
    global image
    im = Image.fromarray(image)
    im.save("CA.png")


def reset():
    array_init()
    visualize()


main_screen.title("Cellular Automata")

empty_label = tk.Label(main_screen, text="",  width=8)
empty_label.grid(row=1, column=2, padx=1, sticky=E)

seeds = tk.Button(main_screen, text="Initialize", command=fill_array, width=12)
seeds.grid(row=1, column=3, padx=1,sticky=E)

all_steps = tk.Button(main_screen, text="Start", command=generate_continuous, width=12)
all_steps.grid(row=1, column=4, padx=1, sticky=E)

sub_step = tk.Button(main_screen, text="Start step by step", command=generate, width=12)
sub_step.grid(row=1, column=5, padx=1,sticky=E)

create_CSV_btt = tk.Button(main_screen, text="Export file", command=export_to_csv, width=12)
create_CSV_btt.grid(row=2, column=3, padx=1,sticky=E)

read_CSV_btt = tk.Button(main_screen, text="Import file", command=import_from_csv, width=12)
read_CSV_btt.grid(row=2, column=4, padx=1,sticky=E)

row_label = tk.Label(main_screen, text="No of rows")
row_label.grid(row=1, column=0, sticky=W)
row = tk.Entry(main_screen, textvariable=row_type, width=7)
row_type.set(rows)
row.grid(row=1, column=1, sticky=W)

col_label = tk.Label(main_screen, text="No of columns")
col_label.grid(row=2, column=0, sticky=W)
col = tk.Entry(main_screen, textvariable=col_type, width=7)
col.grid(row=2, column=1, sticky=W)

grain_label = tk.Label(main_screen, text="No of grains")
grain_label.grid(row=3, column=0, sticky=W)
grain = tk.Entry(main_screen, textvariable=grain_type, width=7)
grain.grid(row=3, column=1, sticky=W)

inclusion_label = tk.Label(main_screen, text="No of inclusions")
inclusion_label.grid(row=4, column=0, sticky=W)
inclusion_no_btt = tk.Entry(main_screen, textvariable= inclusion_type, width=7)
inclusion_no_btt.grid(row=4, column=1, sticky=W)


methods = tk.Listbox(main_screen, height=6)
methods.grid(row=6, column=0, columnspan=2, pady=13)
for key, item in KERNEL.items():
    methods.insert(END, item[0])

visualize_btt = tk.Button(main_screen, text="Visualize", command=visualize, width=12)
visualize_btt.grid(row=2, column=5, padx=1,sticky=E)

write_img_btt = tk.Button(main_screen, text="Export image", command=image_to_png, width=12)
write_img_btt.grid(row=3, column=5, padx=1,sticky=E)

reset_btt = tk.Button(main_screen, text="Reset", command=reset, width=12)
reset_btt.grid(row=7, column=5)

wrap_label = tk.Label(main_screen, text="wrapped array")
wrap_label.grid(row=7, column=0)
wrap_ckbox_var = IntVar()
wrap_ckbox = tk.Checkbutton(main_screen, variable=wrap_ckbox_var)
wrap_ckbox.grid(row=7, column=1)


while 1:
    if start == 1:
        generate()
    main_screen.update_idletasks()
    main_screen.update()














import matplotlib.pyplot as plt
import numpy as np

cur_points_list = np.array([[0.40111595, 0.26086012],
  [0.4155249,  0.31910425],
  [0.34757337, 0.33591467],
  [0.39560312, 0.5300619 ],
  [0.33735895, 0.54447085],
  [0.2893292,  0.35032362],
  [0.22137767, 0.367134  ],
  [0.20696874, 0.30888987]])

tar_points_list = np.array([[0.20627218, 0.22470579],
  [0.24240805, 0.17680797],
  [0.29828882, 0.21896648],
  [0.4187417,  0.05930713],
  [0.46663952, 0.095443  ],
  [0.34618664, 0.25510234],
  [0.40206742, 0.29726085],
  [0.36593154, 0.34515867]])


def main():
    print(cur_points_list.shape)

    plt.xlim((0, 0.6))
    plt.ylim((0, 0.6))

    plt.scatter(cur_points_list[:, 0], cur_points_list[:, 1])
    plt.scatter(tar_points_list[:, 0], tar_points_list[:, 1])

    plt.show()

if __name__ == "__main__":
    main()
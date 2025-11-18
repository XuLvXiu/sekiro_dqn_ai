#encoding=utf8

import cv2
import numpy as np

class Posture(): 
    def __init__(self, blood_gray_min=100, blood_gray_max=130): 
        self.blood_gray_min = blood_gray_min
        self.blood_gray_max = blood_gray_max
        self.debug_name = 'test'

    def detect(self, file_path): 
        print('-' * 100)
        print(file_path)
        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if True: 
            # posture bar is symmetrical, we only use the first half, but the reserved one.
            middle_row = np.flip(gray[gray.shape[0] // 2, 0:gray.shape[1] // 2])
            print('middle_row: ', middle_row)
            self.full_count = len(middle_row)
            first_pixel = middle_row[0]

            if first_pixel >= 230: 
                arr_index = np.where(middle_row > self.blood_gray_max)[0]
                self.hp_count = len(arr_index)
                print(self.hp_count, self.full_count)
                self.status = (self.hp_count / self.full_count) * 100
                return

            if not first_pixel in [183, 184, 185]: 
                self.hp_count = 0
                self.status = 0
                print('first_pixel error:', first_pixel)
                # cv2.imwrite('self.gray.0.%s.png' % (self.debug_name), self.gray)
                return

            # how to check if there is a posture bar?
            # print(self.debug_name)
            arr_index = np.where(middle_row < self.blood_gray_min)[0]
            if arr_index.size == 0: 
                arr_index = np.where(middle_row > self.blood_gray_max)[0]
                if arr_index.size > 15: 
                    self.hp_count = 0
                    self.status = 0
                    # cv2.imwrite('self.gray.000.%s.png' % (self.debug_name), self.gray)
                    # sys.exit(0)
                    return

                self.hp_count = self.full_count
                self.status = 100
                return

            self.hp_count = arr_index[0]
            self.status = (self.hp_count / self.full_count) * 100


# main
p = Posture()
for i in range(0, 70): 
    p.detect('./images/posture_%s.png' % (i))
    print(p.status)


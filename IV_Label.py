'''
This is the software program for labelling
e mode:
p mode:
r mode:
The loop:
'''

import cv2
import numpy as np
import glob
import time
import matplotlib.pyplot as plt

class IV_UI():

    def __init__(self):
        self.Folder = '/home/mgs/PycharmProjects/IV_Vessel/IV_Label/'
        self.imgpath = self.Folder + 'labelread/1.jpg'
        self.labelread = self.Folder + 'labelread/'
        self.labelsave = self.Folder + 'labelsave/'
        self.labelsavenpy = self.Folder + 'labelsavenpy/'

    def PtsFromRec(self):

        P_rec = self.Operation()
        x_rec_1 = P_rec[0][0]
        y_rec_1 = P_rec[0][1]
        x_rec_2 = P_rec[1][0]
        y_rec_2 = P_rec[1][1]
        mask = np.zeros(img_grey.shape, dtype=np.uint8)
        mask[y_rec_1:y_rec_2, x_rec_1:x_rec_2] = 255
        masked_image = cv2.bitwise_and(img_grey, mask)
        cv2.imshow('masked image', masked_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def PtsFromEdge(self):

        P_ROI = self.Operation()
        print("The ROI is ", P_ROI)

        points = np.array([P_ROI], dtype=np.int32)
        points_circle = P_circle
        pixel_use = np.array(points_circle, dtype=np.int32)
        pixel_use = pixel_use[1:, :]
        points = pixel_use
        h = img.shape[0]
        w = img.shape[1]
        mask = np.zeros(img_grey.shape, dtype=np.uint8)
        for item in pixel_use:
            mask[item[0], item[1]] = 255
        masked_image = cv2.bitwise_and(img_grey, mask)
        mask_idx = np.where(masked_image != 0)
        x_roi = mask_idx[1]
        y_roi = mask_idx[0]
        ROI = np.zeros([len(x_roi), 2])
        ROI[:, 0] = x_roi
        ROI[:, 1] = y_roi
        ROI = np.array(ROI, dtype=np.int32)
        pos_seed = np.mean(ROI, 0)
        cv2.imshow('masked image', masked_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def PtsFromCen(self):

        # This is the coordinate from the centroid -- this is important
        mode_user = 'p'
        P_ROI = self.Operation(mode_user)
        print("The centroid of the point in the vessel is ", P_ROI)

        return P_ROI[0]

    def draw_circle(self, event, x, y, flags, param):

        # ix and iy are usually the initialized points
        global ix, iy, drawing, mode, P_circle, P_ROI, P_centroid, P_edge

        # if mode == 'r':
        #     drawing = False
        #     (x, y, w, h) = cv2.selectROI(img)
        #     ix, iy = x, y
        #     x = ix + w
        #     y = iy + h
        #     P_rec.append([ix, iy, x, y])
        #     # print(x, y, ix, iy)

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:

            if drawing == True:

                if mode == 'r':
                    cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3)
                    a = x
                    b = y
                    if a != x | b != y:
                        cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
                if mode == 'e':
                    P_ROI.append([x, y])
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                elif mode == 'p':
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == 'r':
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                P_rec.append([ix, iy, x, y])
            if mode == 'p':
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                P_centroid.append([x, y])
            if mode == 'e':
                cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                P_ROI.append([x, y])

    def Operation(self, mode_user):

        global drawing, mode, img, img_grey, xv, yv, P_ROI, P_circle, P_rec, P_centroid, P_edge
        drawing = False
        mode = True
        ix, iy = -1, -1
        P_ROI = []
        P_circle = np.zeros((1, 2))
        P_rec = []
        P_centroid = []
        P_edge = []
        flag_use = False

        while(flag_use == False):

            # Load the image
            imgpath = self.imgpath
            img = cv2.imread(imgpath, 1)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            [h, w] = img_grey.shape

            # Generate the pixel grid
            x = np.linspace(0, h - 1, h)
            y = np.linspace(0, w - 1, w)
            xv, yv = np.meshgrid(y, x)

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.draw_circle)

            while(1):
                k = cv2.waitKey(1) & 0xFF
                mode = mode_user
                cv2.imshow('image', img)
                if k == 27:
                    break
                elif k == ord('a'):
                    P_centroid = []
                    P_edge = []
                    P_ROI = []
                    P_rec = []
                    print("Relabel this image")
                    print(P_edge)
                    flag_use = False
                    break
                elif k == ord('q'):
                    flag_use = True
                    break

            cv2.destroyAllWindows()

        if mode == 'r':
            return P_rec
        if mode == 'e':
            return P_ROI
        if mode == 'p':
            return P_centroid

    def LABEL(self):

        # Basic setting
        RES = []        # Record the centroid coordinates
        NAME = []       # Record the name of the object
        count = 0

        # Check the data and the folder
        mode_user = input("Which mode you want p(centroid), e(edge), r(rectangular) (p/e/r)")
        input('Please label from left to right for the multi-vessel models (press enter)')
        input('Make sure that you change the name of the input folder (press enter)')
        input('Make sure that you change the name of save folder (press 1/0)')
        listimgname = glob.glob(self.labelread + '*.jpg')
        N_img = len(listimgname)
        print('There are ', str(N_img) + ' images for processing')

        for i in range(int(N_img)):
            print('The ' + str(i+1) + ' image')
            filename = self.labelread + str(i+1) + '.jpg'
            self.imgpath = filename
            img = cv2.imread(filename, 0)
            P_res = self.Operation(mode_user)
            print("The current result is ", P_res)

            # Only for "e" edge mode
            if mode_user == 'e':
                EDGE_use = np.zeros((1, 2))
                for item in P_res:
                    EDGE_use = np.vstack([EDGE_use, item])
                np.save(self.labelsavenpy + str(i + 1) + '.npy', EDGE_use)

                # Show the feature on the image for checking
                for i in range(len(P_res)):
                    cv2.circle(img, tuple(P_res[i]), 8, (0, 0, 255), -1)

            if mode_user == 'p':
                for i in range(len(P_res)):
                    cv2.circle(img, tuple(P_res[i]), 8, (0, 255, 0), -1)

            if mode_user == 'r':
                for i in range(len(P_res)):
                    cv2.rectangle(img, (P_res[i][0], P_res[i][1]), (P_res[i][2], P_res[i][3]), (255, 0, 0), 5)

            # Record the data
            NAME.append(filename)
            RES.append(P_res)

            # Show the image for checking
            plt.imshow(img, cmap = 'gray')
            filename_save = self.labelsave + 'label_' + str(count) + '.jpg'
            cv2.imwrite(filename_save, img)
            count += 1
            plt.show()

        np.save(self.labelsave + 'NAME_invivo.npy', NAME)
        np.save(self.labelsave + 'CEN_invivo.npy', RES)

if __name__ == "__main__":
    test = IV_UI()
    test.LABEL()
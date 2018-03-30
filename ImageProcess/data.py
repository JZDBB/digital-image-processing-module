import os
import cv2
import numpy as np

class Data(object):

    def read_images(self, path, mode):
        images = []
        filenames = os.listdir(path=path)
        for filename in filenames:
            pic_path = os.path.join(path, filename)
            if mode == None:
                images.append(cv2.imread(pic_path))
            else:
                images.append(cv2.imread(pic_path, mode))

            # if type:
            #     # ajust its size to a Regulated template
            #     ratio0 = 1.0
            #     if img.shape[1] < 2500:
            #         ratio0 = img.shape[1] / 2500.0
            #         img = cv2.resize(img, (2500, int(2500 * img.shape[0] / img.shape[1])))
            #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # else:
            #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return images

    def write_images(self, path, name, imgs, rectangles):
        count = 1
        str = "1.png"
        for img in imgs:
            for rect in rectangles:
                pts = np.float32(rect).reshape(-1, 1, 2)
                perspectiveM = cv2.getPerspectiveTransform(rect, pts)
                found = cv2.warpPerspective(img, perspectiveM, (rect[2]-rect[0], rect[3]-rect[1]))
                cv2.imwrite(os.path.join(path, str.replace("1", "%s" % str(name*10+count))), found)
                count = count + 1



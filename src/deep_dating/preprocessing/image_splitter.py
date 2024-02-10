
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageSplitter:

    def __init__(self, patch_size=512, force_size=True, plot=True):
        self.patch_size = patch_size
        self.force_size = force_size
        self.plot = plot

    def _calc_dim(self):
        self.height, self.width = self.img.shape
        
        self.min_dim = min(self.height, self.width)
        self.max_dim = max(self.height, self.width)

        self.extend_width = self.height < self.width

    def _pad_image(self, img, new_shape, extend_width, fill_value=0):
        height, width = img.shape
        new_height, new_width = new_shape
        img_pad = np.zeros(new_shape)
        img_pad.fill(fill_value)

        diff = new_width - width if extend_width else new_height - height
        num_padding = np.floor(diff / 2.0).astype(int)

        if extend_width:
            img_pad[:, num_padding:width+num_padding] = img
        else:
            img_pad[num_padding:height+num_padding, :] = img

        return img_pad.copy()
    
    def _calc_ratio(self):
        if self.force_size:
            self.factor = 1
            self.ratio = self.patch_size / self.min_dim
        else:
            self.factor = int(self.min_dim / self.patch_size)
            self.ratio = (self.factor * self.patch_size) / self.min_dim

    def split(self, img_path):
        self.img_org = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)

        self._calc_dim()

        self._calc_ratio()
        self.img = cv2.resize(self.img, (0, 0), fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_AREA)
        self._calc_dim() # re-calculate dimensions after resize

        num_patches = int(np.ceil(self.max_dim / self.patch_size))
        new_dim = num_patches * self.patch_size

        new_shape = np.array(self.img.shape)
        new_shape[np.argmax(new_shape)] = new_dim
        new_height, new_width = tuple(new_shape)

        if self.plot:
            self.fig = plt.figure()
        
        diff = new_width - self.width if self.extend_width else new_height - self.height
        num_padding = np.floor(diff / 2.0).astype(int)

        patch_start_idx = 0
        patch_end_idx = self.patch_size

        patch_ls = []

        for i in range(self.factor):

            img_split = self.img[patch_start_idx:patch_end_idx, :] if self.extend_width else self.img[:, patch_start_idx:patch_end_idx]

            start_idx = 0
            end_idx = -num_padding + self.patch_size

            for j in range(num_patches):
                
                img = img_split[:, start_idx:end_idx] if self.extend_width else img_split[start_idx:end_idx, :]
                img = self._pad_image(img, (self.patch_size, self.patch_size), extend_width=self.extend_width)
                patch_ls.append(img)

                start_idx = end_idx
                end_idx += self.patch_size

                if self.plot:
                    patch_num = ((i * num_patches) + j) + 1
                    ax = self.fig.add_subplot(self.factor, num_patches, patch_num)
                    ax.imshow(img, cmap="gray")
                    ax.set_title(f"Patch {patch_num}")

            patch_start_idx = patch_end_idx
            patch_end_idx += self.patch_size

        return patch_ls


if __name__ == "__main__":
    image_splitter = ImageSplitter(plot=True)
    path = "../../../../datasets/ICDAR2017_CLaMM_Training/IRHT_P_001274.tif"
    #path = "../../../../datasets/MPS/Download/1550/MPS1550_0024.ppm"
    import glob

    imgs = glob.glob("../../../../datasets/ICDAR2017_CLaMM_Training/*.tif")
    for path in imgs:
        image_splitter.split(path)
        plt.show()
    

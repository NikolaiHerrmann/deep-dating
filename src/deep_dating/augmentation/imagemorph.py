
import numpy as np
from PIL import Image
import imagemorph_cython

class ImageMorph:

    def read_img(self, img_path):
        image = Image.open(img_path).convert("RGB")
        pixels = np.array(image)
        w, h = image.size
        return pixels, w, h
    
    def write_img(self, pixels, output_img_path):
        img = Image.fromarray(pixels, "RGB")
        img.save(output_img_path)
    
    def apply_rubber_sheet(self, img_path, amp, sigma, output_img_path="output.jpg"):
        pixels, w, h = self.read_img(img_path)

        if sigma > h / 2.5 or sigma > w / 2.5:
            print("Warning: Gaussian smoothing kernel too large for the input image.")
            return

        if sigma < 1E-5:
            print("Warning: Gaussian smoothing kernel with negative/zero spread.")
            return
        
        d_x, d_y = self.compute_displacement_field(amp, sigma, h, w)
        output_pixels = self.apply_displacement_field(pixels, d_x, d_y, h, w)

        self.write_img(output_pixels, output_img_path)

    def compute_displacement_field(self, amp, sigma, h, w):
        d_x = (np.random.rand(h, w) * 2) - 1
        d_y = (np.random.rand(h, w) * 2) - 1

        da_x = np.zeros((h, w))
        da_y = np.zeros((h, w))

        kws = int(2.0 * sigma)
        ker = np.exp(-np.square(np.arange(-kws, kws + 1)) / np.square(sigma))

        # for i in range(h):
        #     for j in range(w):
        #         d_x[i][j] = -1.0 + 2.0 * random.random()
        #         d_y[i][j] = -1.0 + 2.0 * random.random()

        d_x, d_y = imagemorph_cython.compute_fields(da_x, da_y, d_x, d_y, h, w, ker, kws)
        avg = np.sum(np.sqrt(np.square(d_x) + np.square(d_y))) / (h * w)

        d_x = np.multiply(d_x, amp)
        d_x = np.divide(d_x, avg)
        
        d_y = np.multiply(d_y, amp)
        d_y = np.divide(d_y, avg)

        return d_x, d_y

    def apply_displacement_field(self, pixels, d_x, d_y, h, w):
        output = np.zeros((h, w, 3), dtype=np.uint8)
        imagemorph_cython.displacement_field(output, pixels, d_x, d_y, h, w)
        return output


if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.random(42)

    im = ImageMorph()
    im.apply_rubber_sheet("MPS1300_0001.ppm", 4, 10)
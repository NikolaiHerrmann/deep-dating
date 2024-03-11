
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from augraphy import BookBinding, PageBorder, BadPhotoCopy, ShadowCast, NoisyLines, BleedThrough, Jpeg, Geometric
from lorem_text import lorem
from deep_dating.util import save_figure, remove_ticks


class AugDoc:

    def __init__(self, plot=True, save_img_dir=None, save_gt_dir=None):
        self.plot = plot
        self.save_img_dir = save_img_dir
        self.save_gt_dir = save_gt_dir
        self.print_count = 1

        self.book_binding = BookBinding(shadow_radius_range=(50, 500),
                                        curve_range_right=(50, 300),
                                        curve_range_left=(50, 300),
                                        curve_ratio_right = (0.05, 0.05),
                                        curve_ratio_left = (0.3, 0.3),
                                        mirror_range=(0.01, 0.01),
                                        binding_align = -1,
                                        binding_pages = (10,10),
                                        curling_direction=-1,
                                        backdrop_color=(255, 255, 255),
                                        enable_shadow=1,
                                        use_cache_images = 0,
                                        )
        self.page_border = PageBorder(page_border_width_height = (30, -40),
                                        page_border_color=(0, 0, 0),
                                        page_border_background_color=(255, 255, 255),
                                        page_border_use_cache_images = 0,
                                        page_border_trim_sides = (0, 0, 0, 0),
                                        page_numbers = "random",
                                        page_rotate_angle_in_order = 0,
                                        page_rotation_angle_range = (1, 10),
                                        curve_frequency=(0, 1),
                                        curve_height=(1, 2),
                                        curve_length_one_side=(30, 60),
                                        same_page_border=0,
                                        )
        self.photo_copy = BadPhotoCopy(noise_type=-1,
                                        noise_side="random",
                                        noise_iteration=(1,1),
                                        noise_size=(1,3),
                                        noise_value=(32, 255),
                                        noise_sparsity=(0.5,0.5),
                                        noise_concentration=(0.99,0.99),
                                        blur_noise=-1,
                                        wave_pattern=0,
                                        edge_effect=0)

        self.shadow_cast = ShadowCast(shadow_side = "random",
                                shadow_vertices_range = (2, 3),
                                shadow_width_range=(0.5, 0.8),
                                shadow_height_range=(0.5, 0.8),
                                shadow_color = (0, 0, 0),
                                shadow_opacity_range=(0.5,0.6),
                                shadow_iterations_range = (1,2),
                                shadow_blur_kernel_range = (101, 301),
                                )

        self.noisy_lines = NoisyLines(noisy_lines_direction = "random",
                                noisy_lines_location = "random",
                                noisy_lines_number_range = (3,5),
                                noisy_lines_color = (0,0,0),
                                noisy_lines_thickness_range = (2,2),
                                noisy_lines_random_noise_intensity_range = (0.01, 0.1),
                                noisy_lines_length_interval_range = (0,100),
                                noisy_lines_gaussian_kernel_value_range = (3,3),
                                noisy_lines_overlay_method = "ink_to_paper",
                                )

        self.bleed_through = BleedThrough(intensity_range=(0.1, 0.15),
                                    color_range=(0, 224),
                                    ksize=(17, 17),
                                    sigmaX=0,
                                    alpha=0.3,
                                    offsets=(10, 20),
                                )

        self.jpeg = Jpeg(quality_range=(60, 95))

        self.geometric = Geometric(scale=(1, 1), translation=(0, 0), flipud=1, crop=(), rotate_range=(1, 10))

        self.fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                      cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]


    def make_img(self, img_name, num_words_range=(1, 20), 
                 img_dim_range=(400, 800), num_lines_range=(3, 10),
                 print_count=0):
        height, width = (random.randint(*img_dim_range), random.randint(*img_dim_range))
        
        font = np.random.choice(self.fonts, size=1)[0]
        background = np.random.randint(128, 255)

        img = np.full((height - 100, width - 100, 3), background, dtype=np.uint8)

        x = 0
        y = 0

        thickness = random.randint(3, 5)
        scale = random.random() + random.randint(1, 2) + 0.5

        for _ in range(random.randint(*num_lines_range)):
            text = lorem.words(random.randint(*num_words_range))
            (x, y) = random.randint(-5, 200), random.randint(y+50,y+100)
            img = cv2.putText(img, text, (x, y), font, scale, 0, thickness)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        img_full = np.full((height, width, 3), background, dtype=np.uint8)

        # Add a small border, so text is not right on edge, will look not real with book bindings
        img_full[50:height-50, 50:width-50] = img

        mask = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
        
        if random.random() > 0.6:
            img_full, mask = self.geometric(img_full, mask=mask)[:2]

        if random.random() > 0.9:
            img_full, mask = self.bleed_through(img_full, mask=mask)[:2]

        if random.random() > 0.3:
            img_full, mask = self.noisy_lines(img_full, mask=mask)[:2]

        if random.random() > 0.4:
            self.page_border.page_numbers = random.randint(5, 50)
            back_color = np.random.randint(1, 128)
            self.page_border.page_border_background_color = (back_color, back_color, back_color)
            img_full, mask = self.page_border(img_full, mask=mask)[:2]
        elif random.random() > 0.3:
            img_full, mask = self.book_binding(img_full, mask=mask)[:2]
        else:
            # no page effects
            pass

        if random.random() > 0.1:
            img_full, mask = self.photo_copy(img_full, mask=mask)[:2]

        if random.random() > 0.1:
            img_full, mask = self.shadow_cast(img_full, mask=mask)[:2]

        if random.random() > 0.5:
            img_full = cv2.flip(img_full, 1)
            mask = cv2.flip(mask, 1)
        
        img_full, mask = self.jpeg(img_full, mask=mask)[:2]
        img_full = cv2.GaussianBlur(img_full, (5, 5), 0)
        
        mask = mask.astype(np.uint8)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Make a border temporarily so floodfill works properly
        mask_height, mask_width = mask.shape
        mask_temp = np.zeros((mask_height + 2, mask_width + 2), dtype=np.uint8)
        mask_temp[1:mask_height+1, 1:mask_width+1] = mask

        mask_temp = cv2.floodFill(mask_temp, None, (0, 0), 255)[1]
        mask = mask_temp[1:mask_height+1, 1:mask_width+1]

        assert mask.shape == img_full.shape[:2], "mask and image are not of same size!"

        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(6, 4))
            ax[0].imshow(img_full)
            ax[0].set_title(f"Synthetic Image {print_count}")
            ax[1].imshow(mask, cmap="gray")
            ax[1].set_title(f"Synthetic Ground Truth {print_count}")

            remove_ticks(ax)
            plt.tight_layout()

        else:
            img_name_save = img_name + ".png"
            gt_name_save = img_name + "_gt.png"

            path_img = os.path.join(self.save_img_dir, img_name_save) if self.save_img_dir else img_name_save
            path_gt = os.path.join(self.save_gt_dir, gt_name_save) if self.save_gt_dir else gt_name_save

            cv2.imwrite(path_img, img_full, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(path_gt, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])



import cv2
import numpy as np
from augraphy import *
import matplotlib.pyplot as plt
from lorem_text import lorem
import random

random.seed(43)

book_binding = BookBinding(shadow_radius_range=(100, 100),
                              curve_range_right=(300, 300),
                              curve_range_left=(200, 200),
                              curve_ratio_right = (0.05, 0.05),
                              curve_ratio_left = (0.3, 0.3),
                              mirror_range=(0.01, 0.01),
                              binding_align = 1,
                              binding_pages = (10,10),
                              curling_direction=-1,
                              backdrop_color=(255, 255, 255),
                              enable_shadow=1,
                              use_cache_images = 0,
                              )
page_border = PageBorder(page_border_width_height = (30, -40),
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
photo_copy = BadPhotoCopy(noise_type=-1,
                                   noise_side="random",
                                   noise_iteration=(1,1),
                                   noise_size=(1,3),
                                   noise_value=(32, 255),
                                   noise_sparsity=(0.5,0.5),
                                   noise_concentration=(0.99,0.99),
                                   blur_noise=-1,
                                   wave_pattern=0,
                                   edge_effect=0)

folding = Folding(fold_count=5,
                  fold_noise=0.0,
                  fold_angle_range = (-360,360),
                  gradient_width=(0.1, 0.2),
                  gradient_height=(0.01, 0.1),
                  backdrop_color = (0,0,0),
                  )
shadow_cast = ShadowCast(shadow_side = "random",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.5,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (101, 301),
                        )

noisy_lines = NoisyLines(noisy_lines_direction = "random",
                        noisy_lines_location = "random",
                        noisy_lines_number_range = (3,5),
                        noisy_lines_color = (0,0,0),
                        noisy_lines_thickness_range = (2,2),
                        noisy_lines_random_noise_intensity_range = (0.01, 0.1),
                        noisy_lines_length_interval_range = (0,100),
                        noisy_lines_gaussian_kernel_value_range = (3,3),
                        noisy_lines_overlay_method = "ink_to_paper",
                        )

letterpress = BleedThrough(intensity_range=(0.1, 0.2),
                            color_range=(0, 224),
                            ksize=(17, 17),
                            sigmaX=0,
                            alpha=0.3,
                            offsets=(10, 20),
                        )

jpeg = Jpeg(quality_range=(60, 95))

geometric = Geometric(scale=(1, 1),
                      translation=(0, 0),
                      flipud=1,
                      crop=(),
                      rotate_range=(-1, 5)
                      )


def make_img(img_name, num_words_range=(1, 20), img_dim_range=(400, 800), num_lines_range=(3, 10)):
 
    height, width = (random.randint(*img_dim_range), random.randint(*img_dim_range))

    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
    
    font = np.random.choice(fonts, size=1)[0]
    background = np.random.randint(128, 255)

    img = np.full((height - 100, width - 100, 3), background, dtype=np.uint8)
    print(img.shape, img.dtype)

    x = 0
    y = 0

    thickness = random.randint(3, 5)
    scale = random.random() + random.randint(1, 2)

    for _ in range(random.randint(*num_lines_range)):
        text = lorem.words(random.randint(*num_words_range))
        (x, y) = random.randint(-5, 200), random.randint(y+50,y+100)
        img = cv2.putText(img, text, (x, y), font, scale, 0, thickness)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    img_full = np.full((height, width, 3), background, dtype=np.uint8)
    img_full[50:height-50, 50:width-50] = img

    mask = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)

    #img_full, mask = folding(img_full, mask=mask)[:2]
    #img_full, mask = geometric(img_full, mask=mask)[:2]
    img_full, mask = letterpress(img_full, mask=mask)[:2]
    img_full, mask = noisy_lines(img_full, mask=mask)[:2]

    page_border.page_numbers = random.randint(5, 50)

    back_color = np.random.randint(1, 128)
    page_border.page_border_background_color = (back_color, back_color, back_color)

    img_full, mask = page_border(img_full, mask=mask)[:2]
    img_full, mask = photo_copy(img_full, mask=mask)[:2]
    img_full, mask = shadow_cast(img_full, mask=mask)[:2]
    img_full, mask = jpeg(img_full, mask=mask)[:2]

    img_full = cv2.GaussianBlur(img_full, (5, 5), 0)
    
    fig, ax = plt.subplots(1, 2)

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.floodFill(mask, None, (0, 0), 255)[1]

    ax[0].imshow(img_full)
    ax[1].imshow(mask, cmap="gray")
    # plt.show()

    cv2.imwrite(img_name + ".png", img_full, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(img_name + "_gt.png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    exit()


while True:
    make_img("None")

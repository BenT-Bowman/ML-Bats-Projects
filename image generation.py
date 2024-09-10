import numpy as np
import random 
import os
import cv2
import argparse
from tqdm import tqdm

def argparse_helper():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--data_path', type=str, required=True, help='Data path. For example ./NewCroppedImages (emergence)')
    parser.add_argument('--save_location', type=str, required=True, help='Path of where to save generated data. For example ./Final Testing Dataset')
    parser.add_argument('--num_2_gen', type=int, default=1000, help='Number of samples to generate.')

    args = parser.parse_args() 
    data_path = args.data_path
    save_location = args.save_location

    return data_path, save_location, args.num_2_gen


def random_point(n)->tuple:
    return np.random.randint(0, n + 1), np.random.randint(0, n + 1)

def too_close(point: tuple, lst: list, thresh: float)->bool:
    """
    Returns True if distance is under thresh, False if it's further
    """
    for other in lst:
        distance = np.linalg.norm(np.array(point) - np.array(other))
        if distance <= thresh:
            return True
    return False

def bat_coords(n, thresh, img_size=40):
    points_list = []
    
    if thresh < 3:
        scaling_value = 0.3
        smallest_point = 20
    elif thresh < 6:
        scaling_value = 0.2
        smallest_point = 25
    elif thresh < 8:
        scaling_value = 0.3
        smallest_point=30
    elif thresh < 11:
        scaling_value = 0.2
        smallest_point =30
    else:
        smallest_point = 35
        scaling_value = 0.1

    width = img_size * scaling_value
    while True:
        attempt_count = 0
        points_list = []
        for _ in range(50):
            p1 = random_point(smallest_point)
            if attempt_count > 5:
                break
            if too_close(p1, points_list, width):
                attempt_count+=1
                continue
            points_list.append(p1)
            if len(points_list)==n:
                return points_list
def mask_with_Kmeans(image, reverse=False):
    """
    If the bat is darker than the background 
    """
    if reverse:
        image = 255-image
    pixel_values = image.reshape((-1, image.shape[-1]))
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(np.float32(pixel_values), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    binary_mask = np.uint8(labels.reshape(image.shape[:2]))

    # Make masking consistant! Takes the mean of the estimated object and background and if the "object" is lighter color swap the two.
    # This assumes that the bats will always be darker than the background, if this is not true
    # invert the image before it is given to this function.
    if cv2.mean(image, mask=binary_mask) > cv2.mean(image, mask=~binary_mask):
        binary_mask = ~binary_mask
    max_val = binary_mask.max()
    min_val = binary_mask.min()
    binary_mask[binary_mask==max_val]=255
    binary_mask[binary_mask==min_val]=0

    return binary_mask

from PIL import Image

def rotate_img(img, theta=45):
    PIL_img = Image.fromarray(img)
    theta_1 = random.uniform(-theta, theta)
    rotated_image = PIL_img.rotate(theta_1)
    return np.asarray(rotated_image)

from PIL import Image, ImageFilter, ImageEnhance
def add_noise(image, factor):
    enhancer = ImageEnhance.Color(image)
    noisy_image = enhancer.enhance(factor)
    return noisy_image

def add_blur(image, radius):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
    return blurred_image

def place_image(img, mask, bkg, coords):
    x, y = coords
    image = Image.fromarray(img)
    masked = Image.fromarray(mask)
    background = Image.fromarray(bkg)
    background.paste(image, (x, y), mask=masked)
    return np.asarray(background)


def shrink_image(img, k):
    image = Image.fromarray(img)
    width, height = image.size
    if k < 4:
        new_width = random.uniform(0.5, 0.8)
    elif k < 8:
        new_width = random.uniform(0.4, 0.6)
    else:
        new_width = random.uniform(0.3, 0.4)


    w = int(width*new_width)
    return np.asarray(image.resize((w, w))), (new_width, new_width)



def load_random(directory):
    files = os.listdir(directory)
    image_path = random.choice(files)
    image_path = os.path.join(directory, image_path)
    image = cv2.imread(image_path)
    return image

# output_folder = 'Final Testing Dataset' # Folder and it's subdirectories must be made prior to running this chunk (cause I'm lazy)
if __name__ == "__main__":
    data_path, output_folder, num_samples_to_generate = argparse_helper()
    print(data_path, output_folder, num_samples_to_generate)
    for i in range(2, 13):
        if not os.path.exists(f"{output_folder}/{i}"):
            os.mkdir(f'{output_folder}/{i}')

    for number_of_bats in range(2, 13):
        for i in tqdm(range(0, num_samples_to_generate)):
            background = load_random(fr'{data_path}\0')
            coords = bat_coords(number_of_bats, number_of_bats)
            for x, y in coords:
                image = load_random(fr'{data_path}\1')
                image, _ = shrink_image(image, number_of_bats)
                masked_img = mask_with_Kmeans(image)
                background = place_image(image, masked_img, background, (x,y))
                
            img_to_save = Image.fromarray(background)
            img_to_save = add_blur(img_to_save, 1)
            img_to_save = img_to_save.convert("L")
            img_to_save.save(fr'{output_folder}\{number_of_bats}\({number_of_bats})_{i}.jpg')
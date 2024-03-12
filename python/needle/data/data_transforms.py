import numpy as np

import needle as ndl


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, _ = img.shape
        shift_x = np.clip(shift_x, -H, H)
        shift_y = np.clip(shift_y, -W, W)
        abs_x = np.abs(shift_x)
        abs_y = np.abs(shift_y)
        start_x_to = max(-shift_x, 0)
        start_y_to = max(-shift_y, 0)
        start_x_from = max(shift_x, 0)
        start_y_from = max(shift_y, 0)
        img_output = np.zeros(img.shape, dtype=img.dtype)
        img_output[start_x_to: start_x_to+H-abs_x, start_y_to: start_y_to+W-abs_y, :] = (
            img[start_x_from: start_x_from+H-abs_x, start_y_from: start_y_from+W-abs_y, :]
        )
        return img_output
        ### END YOUR SOLUTION

import torch
from torchvision.io import read_image
from torchvision.transforms import Resize

T_BBOX = list[list[list[int, int, int, int]]]


def images_bboxes_resizer(
        images: "torch.Tensor",
        bboxes: T_BBOX,
        size: tuple[int, int]
) -> tuple["torch.Tensor", T_BBOX]:
    resized_images = torch.empty(len(images), 3, *size)
    resized_bboxes = []
    for i, (image, bbox) in enumerate(zip(images, bboxes)):
        image = read_image(image)
        x_factor, y_factor = size[0] / image.shape[1], size[1] / image.shape[0]
        resized_images[i] = Resize(size)(image)
        resized_bboxes.append([list(map(int, [a * x_factor, b * y_factor, c * x_factor, d * y_factor]))
                               for a, b, c, d in bbox])
    return resized_images, resized_bboxes


__all__ = [
    "images_bboxes_resizer"
]

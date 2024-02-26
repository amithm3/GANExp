import argparse
import os
import shutil

import imageio
import matplotlib as mpl
import numpy as np
import tensorflow as tf


def arguments():
    parser = argparse.ArgumentParser(description='Create a gif from tensorboard file images')
    parser.add_argument(
        'filename',
        type=str,
        help='Path to tensorboard file'
    )
    parser.add_argument(
        'tag',
        type=str,
        help='Name of the image in the tensorboard e.g. `test/image`.'
    )
    parser.add_argument(
        '--output',
        default="./out.gif",
        type=str,
        help='File to store the final result'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='First image in the gif (corresponds to tensorboard step)'
    )
    parser.add_argument(
        '--stop',
        type=int,
        help='Last image in the gif (corresponds to tensorboard step)'
    )
    return parser.parse_args()


def remove(path):
    """ remove directory or file `path` """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def save_images_from_event(fn, tag, start=-1, stop=np.inf, output_dir="./"):
    assert os.path.isdir(output_dir)

    filenames = []
    for e in tf.compat.v1.train.summary_iterator(fn):
        for v in e.summary.value:
            if v.tag == tag:
                if start < e.step < stop:
                    img = mpl.image.imread(v.tensor.string_val[0])
                    fn = os.path.join(output_dir, f"{e.step}.png")
                    mpl.image.imsave(fn, img)
                    filenames.append(fn)
    return filenames


def image_list_to_gif(filenames, output_fn):
    kwargs = {'duration': .5}
    with imageio.get_writer(output_fn, mode='I', **kwargs) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    args = arguments()
    tmp_output_dir = "/tmp/image_to_gif/"

    if not os.path.isdir(tmp_output_dir):
        os.mkdir(tmp_output_dir)

    names = save_images_from_event(
        args.filename,
        args.tag,
        start=args.start,
        stop=args.stop,
        output_dir=tmp_output_dir
    )

    image_list_to_gif(names, args.output)
    # remove(tmp_output_dir)
    print("GIF CREATED:", args.output)

import os
import sys
import glob
import argparse
import time
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from .__version__ import __version__


MAX_VALUES_BY_DTYPE = {
    "uint8": 255,
    "uint16": 65535,
}


class BasicImageEDA:

    @staticmethod
    def get_img_paths(data_dir, extensions):
        extensions = [''.join(['*', ext]) if ext[0] == '.' else ''.join(['*.', ext]) for ext in extensions]

        img_paths = []
        for extension in extensions:
            img_paths.extend(glob.glob(os.path.join(data_dir, '**', extension), recursive=True))

        return img_paths

    @staticmethod
    def get_img_info(img_path, nonzero, hw_division_factor, channel_hist):

        # read
        if img_path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
            with open(img_path.encode("utf-8"), "rb") as f:
                bytes = bytearray(f.read())
            np_array = np.asarray(bytes)
            img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED).squeeze()
        elif img_path.split('.')[-1].lower() in ['tif', 'tiff']:
            img = skimage.io.MultiImage(img_path)[-1]

        if img is None:  # if image cannot be read
            return None

        else:
            info_dict = {}
            info_dict['extension'] = img_path.split('.')[-1]
            info_dict['dtype'] = str(img.dtype)
            dtype_max = MAX_VALUES_BY_DTYPE[str(img.dtype)]

            info_dict['h'] = img.shape[0]
            info_dict['w'] = img.shape[1]

            if hw_division_factor != 1.0:
                img = cv2.resize(img, None, fx=1/hw_division_factor, fy=1/hw_division_factor, interpolation=cv2.INTER_LINEAR)

            if img.ndim == 2:
                info_dict['channel'] = 1
                img = img.reshape(-1)
            elif img.ndim == 3:
                if img.shape[2] == 3:
                    info_dict['channel'] = 3
                    if img_path.split('.')[-1] in ['png', 'jpg', 'jpeg']:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    info_dict['channel'] = 4
                    if img_path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img_path.split('.')[-1].lower() in ['tif', 'tiff']:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:
                    raise Exception('Only 2-D images with 1,3,4 channels are supported.')

                img = img.reshape(-1, 3)

            if nonzero:
                if info_dict['channel'] == 1:
                    nonzeros = np.where(img != 0)
                    img = img[nonzeros]
                else:
                    nonzeros = np.any(img != [0, 0, 0], axis=-1)
                    img = img[nonzeros]

            if channel_hist:
                rgb_count = np.zeros((3, dtype_max+1), np.int64)

                if info_dict['channel'] == 1:
                    rgb_count[:] = np.histogram(img[:], bins=dtype_max+1, range=[0, float(dtype_max+1)])[0]
                else:
                    for c in range(3):
                        hist = np.histogram(img[:, c], bins=dtype_max+1, range=[0, float(dtype_max+1)])[0]
                        rgb_count[c] = hist
                info_dict['rgb_count'] = rgb_count

            img = img / np.array([dtype_max], dtype=np.float32)

            info_dict['mean'] = img.mean(axis=0)
            info_dict['std'] = (img ** 2).mean(axis=0)

            if info_dict['channel'] == 1:
                info_dict['mean'] = np.repeat(info_dict['mean'], 3)
                info_dict['std'] = np.repeat(info_dict['std'], 3)

            return info_dict

    @classmethod
    def explore(cls, data_dir, extensions=('png', 'jpg', 'jpeg'), threads=0, dimension_plot=True, channel_hist=False,
                nonzero=False, hw_division_factor=1.0):
        """
        Explore image dataset directory to check basic information of the images.

        Args:
            data_dir: str
                image dataset directory path. images are found recursively.
            extensions: array_like
                target image extensions.
            threads: int
                number of multiprocessing threads. if zero, automatically counted.
            dimension_plot: bool
                if True, show dimension(height/width) scatter plot.
            channel_hist: bool
                if True, show channelwise pixel value histogram. takes longer time.
            nonzero: bool
                if True, calculate values only from non-zero pixels of the images.
            hw_division_factor: float
                if a float value other than 1.0 is given, divide height,width of the images by this factor
                to make pixel value calculation faster. Height, width information are not changed and will be printed
                correctly.

        Returns:
            A dict containing information of the images.
        """

        start = time.time()

        img_paths = cls.get_img_paths(data_dir, extensions)
        if len(img_paths) == 0:
            print('No images found.')
            sys.exit()
        else:
            print('\nfound', len(img_paths), 'images.')

        # infos to print out
        ###################################
        output = {
            'dtype': '',
            'channels': [],
            'extensions': [],
            'min_h': float('inf'),
            'max_h': 0,
            'mean_h': 0,
            'min_w': float('inf'),
            'mean_w': 0,
            'max_w': 0,
            'mean_hw_ratio': 0,
            'rec_hw_size_8': np.zeros(2, np.int32),
            'rec_hw_size_16': np.zeros(2, np.int32),
            'rec_hw_size_32': np.zeros(2, np.int32),
            'mean': np.zeros(3, np.float32),
            'std': np.zeros(3, np.float32),
        }

        if dimension_plot:
            heights = []
            widths = []

        cnt = 0
        ###################################

        if threads < 0:
            raise Exception('Number of threads must be at least 1')
        elif threads == 0:
            threads = cpu_count()

        pool = Pool(threads)
        print('Using %d threads.\n' % threads)

        try:
            func = partial(cls.get_img_info, nonzero=nonzero, hw_division_factor=hw_division_factor, channel_hist=channel_hist)
            for info in tqdm(pool.imap_unordered(func, img_paths), total=len(img_paths)):
                if info is None:
                    continue

                if output['dtype'] == '':
                    output['dtype'] = info['dtype']
                else:
                    if output['dtype'] != info['dtype']:
                        raise Exception('different data types are mixed. please use single data type only(uint8 or uint16).')

                output['channels'].append(info['channel'])
                output['extensions'].append(info['extension'])

                h = info['h']
                w = info['w']

                if h < output['min_h']:
                    output['min_h'] = h
                if h > output['max_h']:
                    output['max_h'] = h
                if w < output['min_w']:
                    output['min_w'] = w
                if w > output['max_w']:
                    output['max_w'] = w

                output['mean_h'] += h
                output['mean_w'] += w

                if dimension_plot:
                    heights.append(h)
                    widths.append(w)
                if channel_hist:
                    if not 'rgb_count' in locals():
                        rgb_count = np.zeros((3, MAX_VALUES_BY_DTYPE[info['dtype']]+1), np.int64)

                    rgb_count += info['rgb_count']

                output['mean'] += info['mean']
                output['std'] += info['std']

                cnt += 1

        except KeyboardInterrupt:
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()
        ##################################################################

        if cnt != len(img_paths):
            print('\nSome images were not processed successfully for some reason.\n')

        else:
            output['channels'] = list(set(output['channels']))
            output['extensions'] = list(set(output['extensions']))

            output['mean_h'] = output['mean_h'] / cnt
            output['mean_w'] = output['mean_w'] / cnt
            output['mean_hw_ratio'] = output['mean_h'] / output['mean_w']

            rec_h_8 = round(output['mean_h'] / 8) * 8
            rec_w_8 = round(output['mean_w'] / 8) * 8
            rec_h_16 = round(output['mean_h'] / 16) * 16
            rec_w_16 = round(output['mean_w'] / 16) * 16
            rec_h_32 = round(output['mean_h'] / 32) * 32
            rec_w_32 = round(output['mean_w'] / 32) * 32

            output['rec_hw_size_8'] = np.array([rec_h_8, rec_w_8])
            output['rec_hw_size_16'] = np.array([rec_h_16, rec_w_16])
            output['rec_hw_size_32'] = np.array([rec_h_32, rec_w_32])

            if np.array_equal(np.unique(output['channels']), [1]):
                output['mean'] = output['mean'][0]
                output['std'] = output['std'][0]
            output['mean'] = output['mean'] / cnt
            output['std'] = np.sqrt(output['std'] / cnt - output['mean'] ** 2)

            print()
            print('*--------------------------------------------------------------------------------------*')
            print('%-25s | ' % 'number of images', cnt)
            print()
            print('%-25s | ' % 'dtype', output['dtype'])
            print('%-25s | ' % 'channels', output['channels'])
            print('%-25s | ' % 'extensions', output['extensions'])
            print()
            print('%-25s | ' % 'min height', output['min_h'])
            print('%-25s | ' % 'mean height', output['mean_h'])
            print('%-25s | ' % 'max height', output['max_h'])
            print()
            print('%-25s | ' % 'min width', output['min_w'])
            print('%-25s | ' % 'mean width', output['mean_w'])
            print('%-25s | ' % 'max width', output['max_w'])
            print()
            print('%-25s | ' % 'mean height/width ratio', output['mean_hw_ratio'])
            print('%-25s | ' % 'recommended input size', output['rec_hw_size_8'], '(h x w, multiples of 8)')
            print('%-25s | ' % 'recommended input size', output['rec_hw_size_16'], '(h x w, multiples of 16)')
            print('%-25s | ' % 'recommended input size', output['rec_hw_size_32'], '(h x w, multiples of 32)')
            print()
            print('%-25s | ' % 'channel mean(0~1)', output['mean'])
            print('%-25s | ' % 'channel std(0~1)', output['std'])
            if hw_division_factor != 1.0:
                print('%-25s | ' % '', 'mean,std were calculated with image heights,widths divided by', hw_division_factor)

            print('*--------------------------------------------------------------------------------------*')
            t = time.time() - start
            print('eda ended in %02d hours %02d minutes %02d seconds'
                  % (t // 3600, (t % 3600) // 60, (t % 3600) % 60))

            if dimension_plot:
                plt.figure('dimension plot', figsize=(8,6))

                plt.axhline(y=output['min_h'], color='orange', linestyle='--', linewidth=1)
                plt.axhline(y=output['max_h'], color='orange', linestyle='--', linewidth=1)
                plt.axvline(x=output['min_w'], color='orange', linestyle='--', linewidth=1)
                plt.axvline(x=output['max_w'], color='orange', linestyle='--', linewidth=1)

                plt.fill_between([output['min_w'], output['max_w']], output['min_h'], output['max_h'],
                                facecolor='orange', alpha=0.1)

                plt.scatter(widths, heights, s=7, c='green')
                plt.scatter(output['mean_w'], output['mean_h'], s=15, c='blue')
                plt.text(output['mean_w'], output['mean_h'], 'mean', color='b', ha='right', va='bottom', fontsize=10)

                axis = plt.axis()
                max_axis = np.max(axis) * 1.05

                if output['min_h'] != output['max_h']:
                    plt.text(max_axis, output['min_h'], ' min %d' % output['min_h'], color='r', ha='left', va='center', fontsize=8)
                    plt.text(max_axis, output['max_h'], ' max %d' % output['max_h'], color='r', ha='left', va='center', fontsize=8)
                else:
                    plt.text(max_axis, output['max_h'], ' fixed height:%d' % output['max_h'], color='r', ha='left', va='center')

                if output['min_w'] != output['max_w']:
                    plt.text(output['min_w'], max_axis, 'min %d' % output['min_w'], color='r', ha='left', va='bottom', rotation=30, fontsize=8)
                    plt.text(output['max_w'], max_axis, 'max %d' % output['max_w'], color='r', ha='left', va='bottom', rotation=30, fontsize=8)
                else:
                    plt.text(output['max_w'], max_axis, 'fixed width:%d' % output['max_w'], color='r', ha='center', va='bottom')

                plt.xlim([0, max_axis])
                plt.ylim([0, max_axis])

                plt.xlabel('width')
                plt.ylabel('height')

                plt.title('height/width scatter plot', pad=30)

                plt.gca().set_aspect('equal', adjustable='box')

            if channel_hist:
                fig = plt.figure('channel histogram', figsize=(10, 6))
                ax = plt.axes([0.1, 0.1, 0.65, 0.8])

                for i, col in enumerate(('r', 'g', 'b')):
                    ax.plot(rgb_count[i], color=col, linewidth=1)

                rax = plt.axes([0.78, 0.35, 0.2, 0.3])
                labels = ['remove_zero', 'remove_maxval']
                check = CheckButtons(rax, labels)

                flags = {'remove_zero': False, 'remove_maxval': False}

                def func(label):
                    flags[label] = not flags[label]

                    new_rgb_count = rgb_count.copy()
                    if flags['remove_zero']:
                        new_rgb_count[:, 0] = 0
                    if flags['remove_maxval']:
                        new_rgb_count[:, -1] = 0

                    ax.cla()

                    for i, col in enumerate(('r', 'g', 'b')):
                        ax.plot(new_rgb_count[i], color=col, linewidth=1)

                    ax.set_xlim([0, rgb_count.shape[1]])
                    ax.set_ylim([0, None])

                    ax.set_xlabel('pixel value')
                    ax.set_ylabel('frequency')

                    ax.set_title('channelwise pixel value histogram (zero omitted)', pad=30)

                    plt.show()

                check.on_clicked(func)

                ax.set_xlim([0, rgb_count.shape[1]])
                ax.set_ylim([0, None])

                ax.set_xlabel('pixel value')
                ax.set_ylabel('frequency')

                ax.set_title('channelwise pixel value histogram (zero omitted)', pad=30)

            if dimension_plot or channel_hist:
                plt.show()

            return output

    @classmethod
    def check_same(cls):
        # To Be Implemented.
        return


def normalize():
    # To Be Implemented.
    return


def main():
    parser = argparse.ArgumentParser(description='image dataset eda tool to check basic infos of images.')
    parser.add_argument('data_dir', type=str,
                        help='image dataset directory path. images are found recursively.')
    parser.add_argument('-e', '--extensions', nargs='+', default=['png', 'jpg', 'jpeg'],
                        help='target image extensions.')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help='number of multiprocessing threads. if zero, automatically counted.')
    parser.add_argument('-d', '--dimension_plot', default=False, action='store_true',
                        help='show dimension(height/width) scatter plot.')
    parser.add_argument('-c', '--channel_hist', default=False, action='store_true',
                        help='show channelwise pixel value histogram. takes much longer time.')
    parser.add_argument('-n', '--nonzero', action='store_true', default=False,
                        help='calculate values only from non-zero pixels of the images.')
    parser.add_argument('-f', '--hw_division_factor', type=float, default=1.0,
                        help='divide height,width of the images by this factor to make pixel value'
                             'calculation faster. Height, width information are not changed and'
                             'will be printed correctly.')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    BasicImageEDA.explore(args.data_dir, args.extensions, args.threads, args.dimension_plot, args.channel_hist,
                          args.nonzero, args.hw_division_factor)


if __name__ == "__main__":
    main()

import os
import sys
import glob
import argparse
import time
import cv2
import skimage.io
from PIL import Image
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

SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'jpe', 'png', 'bmp', 'ppm', 'pbm', 'pgm', 'sr', 'ras',  # opencv
                        'tif', 'tiff',  # skimage.io
                        'webp'  # PIL
                        ]


class BasicImageEDA:

    @staticmethod
    def get_img_paths(data_dir, extensions):
        if extensions is None:
            extensions = SUPPORTED_EXTENSIONS

        extensions = [''.join(['*', ext]) if ext[0] == '.' else ''.join(['*.', ext]) for ext in extensions]

        img_paths = []
        for extension in extensions:
            img_paths.extend(glob.glob(os.path.join(data_dir, '**', extension), recursive=True))

        return img_paths

    @staticmethod
    def get_img_info(img_path, nonzero, hw_division_factor, channel_hist):

        # read
        if img_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'jpe', 'png', 'bmp', 'ppm', 'pbm', 'pgm', 'sr', 'ras']:
            with open(img_path.encode("utf-8"), "rb") as f:
                bytes = bytearray(f.read())
            np_array = np.asarray(bytes)
            img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED).squeeze()
        elif img_path.split('.')[-1].lower() in ['tif', 'tiff']:
            # uses the last image of tiff multi image. temporary.
            img = skimage.io.MultiImage(img_path)[-1]
        elif img_path.split('.')[-1].lower() in ['webp']:
            img = Image.open(img_path).convert('RGB')
            img = np.asarray(img)

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
                    if img_path.split('.')[-1] in ['jpg', 'jpeg', 'jpe', 'png', 'bmp', 'ppm', 'pbm', 'pgm', 'sr', 'ras']:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    info_dict['channel'] = 4
                    if img_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'jpe', 'png', 'bmp', 'ppm', 'pbm', 'pgm', 'sr', 'ras']:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img_path.split('.')[-1].lower() in ['tif', 'tiff', 'webp']:
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

    @staticmethod
    def show_dimension_plot(info_dict, widths, heights):

        plt.figure('dimension plot', figsize=(8, 6))

        plt.axhline(y=info_dict['min_h'], color='orange', linestyle='--', linewidth=1)
        plt.axhline(y=info_dict['max_h'], color='orange', linestyle='--', linewidth=1)
        plt.axvline(x=info_dict['min_w'], color='orange', linestyle='--', linewidth=1)
        plt.axvline(x=info_dict['max_w'], color='orange', linestyle='--', linewidth=1)

        plt.fill_between([info_dict['min_w'], info_dict['max_w']], info_dict['min_h'], info_dict['max_h'],
                         facecolor='orange', alpha=0.1)

        plt.scatter(widths, heights, s=7, c='green')
        plt.scatter(info_dict['mean_w'], info_dict['mean_h'], s=15, c='blue')
        plt.text(info_dict['mean_w'], info_dict['mean_h'], 'mean', color='b', ha='right', va='bottom', fontsize=10)

        axis = plt.axis()
        max_axis = np.max(axis) * 1.05

        if info_dict['min_h'] != info_dict['max_h']:
            plt.text(max_axis, info_dict['min_h'], ' min %d' % info_dict['min_h'], color='r', ha='left', va='center',
                     fontsize=8)
            plt.text(max_axis, info_dict['max_h'], ' max %d' % info_dict['max_h'], color='r', ha='left', va='center',
                     fontsize=8)
        else:
            plt.text(max_axis, info_dict['max_h'], ' fixed height:%d' % info_dict['max_h'], color='r', ha='left', va='center')

        if info_dict['min_w'] != info_dict['max_w']:
            plt.text(info_dict['min_w'], max_axis, 'min %d' % info_dict['min_w'], color='r', ha='left', va='bottom',
                     rotation=30, fontsize=8)
            plt.text(info_dict['max_w'], max_axis, 'max %d' % info_dict['max_w'], color='r', ha='left', va='bottom',
                     rotation=30, fontsize=8)
        else:
            plt.text(info_dict['max_w'], max_axis, 'fixed width:%d' % info_dict['max_w'], color='r', ha='center', va='bottom')

        plt.xlim([0, max_axis])
        plt.ylim([0, max_axis])

        plt.xlabel('width')
        plt.ylabel('height')

        plt.title('height/width scatter plot', pad=30)

        plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()

    @staticmethod
    def show_channel_hist(rgb_count):
        # takes longer time than show_dimension_plot()

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

            ax.set_title('channelwise pixel value histogram', pad=30)

            plt.show()

        check.on_clicked(func)

        ax.set_xlim([0, rgb_count.shape[1]])
        ax.set_ylim([0, None])

        ax.set_xlabel('pixel value')
        ax.set_ylabel('frequency')

        ax.set_title('channelwise pixel value histogram', pad=30)

        plt.show()

    @classmethod
    def explore(cls, data_dir, extensions=None, threads=0, dimension_plot=False, channel_hist=False,
                nonzero=False, hw_division_factor=1.0):
        """
        Explore image dataset directory to check basic information of the images.

        Args:
            data_dir: str
                image dataset directory path. images are found recursively.
            extensions: array_like
                target image extensions.
            threads: int
                number of multiprocessing threads. if zero, automatically count max threads.
            dimension_plot: bool
                if True, show dimension(height/width) scatter plot.
            channel_hist: bool
                if True, show channelwise pixel value histogram. takes longer time.
            nonzero: bool
                if True, calculate values only from non-zero pixels of the images.
            hw_division_factor: float
                if a float value other than 1.0 is given, divide height,width of the images by this factor
                to make pixel value calculation faster. Information on height, width are not changed and
                will be printed correctly.

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

        # to print out
        ###################################
        output = {
            'dtype': '',
            'channels': [],
            'extensions': [],
            'min_h': 0,
            'max_h': 0,
            'mean_h': 0,
            'median_h': 0,
            'min_w':0,
            'max_w': 0,
            'mean_w': 0,
            'median_w': 0,
            'mean_hw_ratio': 0,
            'median_hw_ratio': 0,
            'rec_hw_size_8': np.zeros(2, np.int32),
            'rec_hw_size_16': np.zeros(2, np.int32),
            'rec_hw_size_32': np.zeros(2, np.int32),
            'mean': np.zeros(3, np.float32),
            'std': np.zeros(3, np.float32),
        }

        cnt = 0

        heights = []
        widths = []
        ###################################

        if threads < 0:
            raise Exception('Number of threads must be at least 1. Use 0 to automatically calculate max threads.')

        max_threads = cpu_count()

        if threads == 0:
            threads = max_threads
        else:
            threads = min(threads, max_threads)

        pool = Pool(threads)
        print('Using %d threads. (max:%d)\n' % (threads, max_threads))

        ##################################################################
        try:  # collect info
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

                heights.append(info['h'])
                widths.append(info['w'])

                output['mean'] += info['mean']
                output['std'] += info['std']

                if channel_hist:
                    if not 'rgb_count' in locals():
                        rgb_count = np.zeros((3, MAX_VALUES_BY_DTYPE[info['dtype']]+1), np.int64)

                    rgb_count += info['rgb_count']

                cnt += 1

        except KeyboardInterrupt:
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()
        ##################################################################

        if cnt != len(img_paths):
            print('\nSome images were not processed successfully for some reason.'
                  'ex)unsupported extension. unicode in file name. \n')

        else:  # manipulation
            output['channels'] = list(set(output['channels']))
            output['extensions'] = list(set(output['extensions']))

            output['min_h'] = min(heights)
            output['max_h'] = max(heights)
            output['mean_h'] = float(np.mean(heights))
            output['median_h'] = int(np.median(heights))

            output['min_w'] = min(widths)
            output['max_w'] = max(widths)
            output['mean_w'] = float(np.mean(widths))
            output['median_w'] = int(np.median(widths))

            output['mean_hw_ratio'] = output['mean_h'] / output['mean_w']
            output['median_hw_ratio'] = output['median_h'] / output['median_w']

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
            print('%-40s | ' % 'number of images', cnt)
            print()
            print('%-40s | ' % 'dtype', output['dtype'])
            print('%-40s | ' % 'channels', output['channels'])
            print('%-40s | ' % 'extensions', output['extensions'])
            print()
            print('%-40s | ' % 'min height', output['min_h'])
            print('%-40s | ' % 'max height', output['max_h'])
            print('%-40s | ' % 'mean height', output['mean_h'])
            print('%-40s | ' % 'median height', output['median_h'])
            print()
            print('%-40s | ' % 'min width', output['min_w'])
            print('%-40s | ' % 'max width', output['max_w'])
            print('%-40s | ' % 'mean width', output['mean_w'])
            print('%-40s | ' % 'median width', output['median_w'])
            print()
            print('%-40s | ' % 'mean height/width ratio', output['mean_hw_ratio'])
            print('%-40s | ' % 'median height/width ratio', output['median_hw_ratio'])
            print('%-40s | ' % 'recommended input size(by mean)', output['rec_hw_size_8'], '(h x w, multiples of 8)')
            print('%-40s | ' % 'recommended input size(by mean)', output['rec_hw_size_16'], '(h x w, multiples of 16)')
            print('%-40s | ' % 'recommended input size(by mean)', output['rec_hw_size_32'], '(h x w, multiples of 32)')
            print()
            print('%-40s | ' % 'channel mean(0~1)', output['mean'])
            print('%-40s | ' % 'channel std(0~1)', output['std'])
            if hw_division_factor != 1.0:
                print('%-40s | ' % '', 'mean,std were calculated with image heights,widths divided by', hw_division_factor)

            print('*--------------------------------------------------------------------------------------*')
            t = time.time() - start
            print('eda ended in %02d hours %02d minutes %02d seconds'
                  % (t // 3600, (t % 3600) // 60, (t % 3600) % 60))

            if dimension_plot:
                cls.show_dimension_plot(output, widths, heights)

            if channel_hist:
                cls.show_channel_hist(rgb_count)

            if dimension_plot or channel_hist:
                plt.show()

            return output

    @classmethod
    def check_same(cls):
        # check if there are same images in the dataset.
        # To Be Implemented.
        return


def main():
    parser = argparse.ArgumentParser(description='image dataset eda tool to check basic infos of images.')
    parser.add_argument('data_dir', type=str,
                        help='image dataset directory path. images are found recursively.')
    parser.add_argument('-e', '--extensions', nargs='+', default=None,
                        help='target image extensions.')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help='number of multiprocessing threads. if zero, automatically count max threads.')
    parser.add_argument('-d', '--dimension_plot', default=False, action='store_true',
                        help='show dimension(height/width) scatter plot.')
    parser.add_argument('-c', '--channel_hist', default=False, action='store_true',
                        help='show channelwise pixel value histogram. takes longer time.')
    parser.add_argument('-n', '--nonzero', action='store_true', default=False,
                        help='calculate values only from non-zero pixels of the images.')
    parser.add_argument('-f', '--hw_division_factor', type=float, default=1.0,
                        help='divide height,width of the images by this factor to make pixel value'
                             'calculation faster. Information on height, width are not changed and'
                             'will be printed correctly.')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s {}'.format(__version__))

    args = parser.parse_args()

    BasicImageEDA.explore(args.data_dir, args.extensions, args.threads, args.dimension_plot, args.channel_hist,
                          args.nonzero, args.hw_division_factor)


if __name__ == "__main__":
    main()

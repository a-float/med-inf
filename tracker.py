import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time
import pickle
import librosa
import librosa.display
import threading
from pygame import mixer
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter


class Inka:
    def __init__(self):
        self.window_len = 10
        self.use3d = False
        self.frames = None
        self.trim_dims = {"top" : 50, "right" : 300, "bottom": 450, "left": 200}
        self.frames_filtered = None
        self.y = None
        self.y2 = None
        self.sr = None
        self.max_step = 1
        self.im_video = None
        self.im_video_proc = None
        self.line_audio = None
        self.line_audio_proc = None
        self.ani = None
        self.surface_ax = None
        self.surface_plot = None
        self.xx, self.yy = np.meshgrid(np.arange(0, 800, 1), np.arange(0, 600, 1))
        self.roi_size = 80//2
        self.roi_cx = 300
        self.roi_cy = self.roi_size
        self.roi_color = (255, 0, 0)
        self.roi_thickness = 2
        self.signal_x = [0]
        self.signal_y = [0]
        self.signal_plot = None

        # --- load data
    def load_audio_and_video(self):
        self.frames, frames_filtered = self.load_video_data()
        pads = self.trim_dims
        shape = frames_filtered.shape
        self.frames_filtered = frames_filtered[:, pads["top"]:shape[1] - pads["bottom"], pads["left"]:shape[2] - pads["right"]]
        self.y, self.sr = self.load_audio_data()

    @staticmethod
    def load_video_data():
        # todo: enter paths to dataset dir and video file 
        path_dir = r"."
        video_filename = r"video1.wmv"

        pickle_filename = video_filename[:-3] + 'p'
        path_video = os.path.join(path_dir, video_filename)
        path_pickle = os.path.join(path_dir, pickle_filename)
        is_pickle = os.path.exists(path_pickle)
        print('video file  ', path_video, ' exists: ', os.path.exists(path_video))
        print('pickle file ', path_pickle, ' exists: ', is_pickle)

        if not is_pickle:
            vid = cv2.VideoCapture(path_video)
            frames = []
            t_start = time.time()
            while True:
                vid.grab()
                retval, frame = vid.retrieve()
                if not retval:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)

            print('loaded frames from video: {}  ({:.2f} s.)'.format(len(frames), (time.time() - t_start)))

            print('gauss filtering...')
            # todo: implement gaussian filtering
            # this should be a list of filtered vieo frames
            frames_filtered = list(map(lambda frame: cv2.GaussianBlur(frame, (7,7), 0), frames))
            print('gauss filtering finished')

            with open(path_pickle, 'wb') as handle:
                pickle.dump((frames, frames_filtered), handle)
            print('pickle saved')
        else:
            t_start = time.time()
            with open(path_pickle, 'rb') as handle:
                (frames, frames_filtered) = pickle.load(handle)
            print('pickle loaded')
            print('loaded frames from pickle: {}  ({:.2f} s.)'.format(len(frames), (time.time() - t_start)))
            print('loaded filtered frames from pickle: {}  ({:.2f} s.)'.format(len(frames_filtered), (time.time() - t_start)))

        return np.array(frames), np.array(frames_filtered)

    @staticmethod
    def load_audio_data():
        # todo: enter path to audio file
        audio_path = r"audio1.wav"
        y, sr = librosa.load(audio_path, sr=16000)
        print('audio data loaded, len: ', len(y))
        return y, sr

    # --- audio processing
    def process_audio(self):
        self.y2 = Inka.butter_bandpass_filter(self.y, 500, 4000, self.sr, order=6)
        self.y2 *= 400      # just more intuitive scale

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        # todo: implement bandpass butterworth filter
        # hint: normalize the low and highcuts using the sampling rate
        low = lowcut / fs
        high = highcut / fs
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = Inka.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def anim_process_frame(self, i):
        if i == 0:
            self.signal_x = []
            self.signal_y = []
        frame = self.frames[i]
        frame_filtered = self.frames_filtered[i]
        start_idx = np.max([0, i - self.window_len])
        end_idx = np.max([1, i])
        # todo: calculate median of frames between start and end idxs
        frame_median = np.median(self.frames_filtered[start_idx:end_idx, :, :], axis=0)

        # todo: calculate the difference between the filtered frames and the median
        # hint: normalize the frame values afterwards
        frame_diff_median = np.abs(frame_filtered - frame_median)
        small, big = np.min(frame_diff_median), np.max(frame_diff_median)
        if big > 0:
            frame_diff_median = 255 * (frame_diff_median - small) / (big - small)
        frame_diff_median = np.uint8(frame_diff_median)
        threshold = 120

        pads = (self.trim_dims["top"], self.trim_dims["bottom"]), (self.trim_dims["left"], self.trim_dims["right"])
        frame_diff_median = np.lib.pad(frame_diff_median, pads, "constant", constant_values=0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame_diff_median_rgb = cv2.applyColorMap(frame_diff_median, cv2.COLORMAP_AUTUMN)
        frame_diff_median_rgb = cv2.cvtColor(frame_diff_median_rgb, cv2.COLOR_BGR2RGB)

        frame_rgb[frame_diff_median > threshold] = 0
        frame_diff_median_rgb[frame_diff_median < threshold] = 0
        frame_rgb += frame_diff_median_rgb

        # todo: change the ROI descriptor
        
        roi_x1 = self.roi_cx - self.roi_size
        roi_x2 = self.roi_cx + self.roi_size 
        roi_y1 = self.roi_cy - self.roi_size
        roi_y2 = self.roi_cy + self.roi_size
        # roi = frame_diff_median[self.roi_x1:self.roi_x2, self.roi_y1:self.roi_y2]
        # moment_roi = frame_diff_median[roi_y1:roi_y2, roi_x1:roi_x2]
        # # already after one thresholding, now just to binarize
        tracking_mask = cv2.erode(frame_diff_median, (3,3), iterations=1)
        tracking_mask = cv2.dilate(tracking_mask, (5,5))
        tracking_mask[tracking_mask < 150] = 0
        roi_moments = cv2.moments(tracking_mask)
        if roi_moments["m00"] != 0:
            blob_cx = int(roi_moments["m10"] / roi_moments["m00"])
            blob_cy = int(roi_moments["m01"] / roi_moments["m00"])
            diff_x = np.clip(blob_cx - self.roi_cx, -self.max_step, self.max_step)
            diff_y = np.clip(blob_cy - self.roi_cy, -self.max_step, self.max_step)
            self.roi_cx += diff_x
            self.roi_cy += diff_y
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_desc = np.sum(roi)/100000
        self.signal_x.append(0.033 * i)
        self.signal_y.append(roi_desc)
        print('roi desc:', roi_desc)

        # --- draw roi
        roi_x1 = self.roi_cx - self.roi_size
        roi_x2 = self.roi_cx + self.roi_size
        roi_y1 = self.roi_cy - self.roi_size
        roi_y2 = self.roi_cy + self.roi_size
        roi_start = (roi_x2 - 1, roi_y1)
        roi_end = (roi_x1, roi_y2 - 1)
        frame_rgb = cv2.rectangle(frame_rgb, roi_start, roi_end, self.roi_color, self.roi_thickness)


        # --- 3d
        if self.use3d:
            frame = frame / 3.0
            frame[frame_diff_median > threshold] = 250
            zz = frame/255.0
            # zz = frame_diff_median/255.0
            self.surface_plot.remove()
            self.surface_plot = self.surface_ax.plot_surface(self.xx, self.yy, zz, linewidth=0, antialiased=False, cmap=cm.plasma)


        tracking_mask_rgb =  cv2.applyColorMap(cv2.cvtColor(tracking_mask, cv2.COLOR_BGR2RGB), cv2.COLORMAP_DEEPGREEN)
        tracking_mask_rgb = cv2.rectangle(tracking_mask_rgb, roi_start, roi_end, self.roi_color, self.roi_thickness)

        # --- set plots
        # im_video.set_array(frame)
        self.im_video.set_array(frame_rgb)
        if not self.use3d:
            # self.im_video_proc.set_array(frame_diff_median)
            self.im_video_proc.set_array(tracking_mask_rgb)
        self.line_audio.set_xdata(0.033 * i)
        self.line_audio_proc.set_xdata(0.033 * i)
        self.signal_plot.set_xdata(self.signal_x)
        self.signal_plot.set_ydata(self.signal_y)
        if not self.use3d:
            return [self.im_video, self.im_video_proc, self.line_audio, self.line_audio_proc, self.signal_plot]
        else:
            return [self.im_video, self.surface_ax, self.line_audio, self.line_audio_proc, self.signal_plot]

    # --- display
    @staticmethod
    def display_spectro(y, sr):
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 8))
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[0])
        ax[0].set(title='Linear-frequency power spectrogram')
        ax[0].label_outer()
        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.show()

    def display_video_and_audio(self):
        initial_frame = self.frames[0]
        fig = plt.figure(1, figsize=(14, 8))
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))

        plt.subplot(2, 2, 1)
        initial_frame_rgb = cv2.cvtColor(initial_frame, cv2.COLOR_GRAY2RGB)
        self.im_video = plt.imshow(initial_frame_rgb, interpolation='none')

        # --- processed video
        if not self.use3d:
            plt.subplot(2, 2, 2)
            self.im_video_proc = plt.imshow(initial_frame_rgb, interpolation='none')

        # --- OR

        # --- surface
        if self.use3d:
            self.surface_ax = fig.add_subplot(2, 2, 2, projection='3d')
            zz = initial_frame/255.0
            self.surface_plot = self.surface_ax.plot_surface(self.xx, self.yy, zz, linewidth=0, antialiased=False, cmap=cm.plasma)

        # --- wave
        plt.subplot(2, 2, 3)
        librosa.display.waveshow(self.y, sr=self.sr)
        self.line_audio = plt.axvline(x=10, ymin=0, ymax=1, color='r', linewidth='1')

        # --- wave processed
        plt.subplot(2, 2, 4)
        librosa.display.waveshow(self.y2, sr=self.sr)
        self.line_audio_proc = plt.axvline(x=0, ymin=0, ymax=1, color='r', linewidth='1')
        self.signal_plot, = plt.plot(self.signal_x, self.signal_y, 'r')

        self.ani = animation.FuncAnimation(fig, self.anim_process_frame,
                                  # fargs=(frames, frame_median, im_video, im_video_proc, line_audio, line_audio_proc),
                                  frames=len(self.frames),
                                  interval=0.05,
                                  blit=True)
        plt.show()


if __name__ == '__main__':
    inka = Inka()
    inka.load_audio_and_video()
    inka.process_audio()
    inka.display_video_and_audio()




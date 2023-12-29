from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os


def read_tdms_channel_data(tdms_file_path, channel_name):
    tdms_file = TdmsFile.read(tdms_file_path)

    channel_data = []
    for group in tdms_file.groups():
        for channel in group.channels():
            if channel.name == channel_name:
                channel_data.append(channel[:])

    return np.concatenate(channel_data)


def plot_and_save_combined(segment_data, sampling_rate, save_dir, segment_idx, oaspl):
    time_axis = np.linspace(0, len(segment_data) / sampling_rate, len(segment_data))

    # 子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 幅度图
    axs[0].plot(time_axis, segment_data)
    axs[0].set_title("Amplitude Plot")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    # 频谱图
    f, t, Sxx = scipy.signal.spectrogram(segment_data, sampling_rate)
    axs[1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title("Spectrogram")

    # 功率谱密度
    f, Pxx = scipy.signal.welch(segment_data, sampling_rate, nperseg=1024)

    # 有PSD值为正
    Pxx[Pxx <= 0] = np.min(Pxx[Pxx > 0])

    # PSD
    axs[2].plot(f, 10 * np.log10(Pxx))
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('PSD (dB/Hz)')
    axs[2].set_title("Power Spectral Density")

    for ax in axs:
        ax.text(0.95, 0.95, f"OASPL: {oaspl:.2f} dB", horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # 保存
    plt.tight_layout()
    plt.savefig(f"{save_dir}/CombinedPlot_{segment_idx}.png")
    plt.close()


def process_segment(data, start_idx, end_idx, num_samples_per_segment, sampling_rate, threshold, save_dir, segment_idx):
    segment_data = data[start_idx:end_idx]
    central_segment_data = data[segment_idx * num_samples_per_segment:(segment_idx + 1) * num_samples_per_segment]
    rms = np.sqrt(np.mean(np.square(central_segment_data)))
    oaspl = 20 * np.log10(rms / 2e-5)

    if oaspl > threshold:
        plot_and_save_combined(segment_data, sampling_rate, save_dir, segment_idx, oaspl)

    return oaspl > threshold, oaspl


def create_save_directory(base_dir, tdms_file_path, threshold):
    file_name = os.path.splitext(os.path.basename(tdms_file_path))[0]

    new_dir_name = f"{file_name}_Threshold_{threshold}"

    new_dir_path = os.path.join(base_dir, new_dir_name)

    # 如果文件夹不存在则创建
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    return new_dir_path


def process_tdms_file(tdms_file_path, channel_name, sampling_rate, segment_length, threshold, base_save_dir):
    save_dir = create_save_directory(base_save_dir, tdms_file_path, threshold)
    data = read_tdms_channel_data(tdms_file_path, channel_name)

    num_samples_per_segment = int(segment_length * sampling_rate)
    extra_samples = int(0.015 * sampling_rate)
    num_segments = len(data) // num_samples_per_segment
    segment_oaspl = []
    num_exceeds_threshold = 0

    for i in range(num_segments):
        start_idx = max(0, i * num_samples_per_segment - extra_samples)
        end_idx = min(len(data), (i + 1) * num_samples_per_segment + extra_samples)
        exceeds_threshold, oaspl = process_segment(data, start_idx, end_idx, num_samples_per_segment, sampling_rate,
                                                   threshold, save_dir, i)
        segment_oaspl.append(oaspl)
        if exceeds_threshold:
            num_exceeds_threshold += 1

    average_oaspl = np.mean(segment_oaspl)
    average_exceeds_per_hour = num_exceeds_threshold / (num_segments * segment_length / 3600)
    return average_oaspl, average_exceeds_per_hour, num_exceeds_threshold


def main(tdms_file_paths, channel_name, sampling_rate, segment_length, oaspl_threshold, base_save_dir):
    for path in tdms_file_paths:
        average_oaspl, average_exceeds_per_hour, num_exceeds_threshold = process_tdms_file(
            path, channel_name, sampling_rate, segment_length, oaspl_threshold, base_save_dir)
        print(f"文件：{path}")
        print("平均OASPL:", average_oaspl)
        print("平均每小时超过阈值的段数:", average_exceeds_per_hour)
        print("超过阈值的段数总计:", num_exceeds_threshold)



"""tdms_file_paths = ["G:\PlantDataV2\dry\dry_yc_4-dark.tdms",
                   "G:\PlantDataV2\dry\dry_yc_4-dark_0.tdms",
                   "G:\PlantDataV2\dry\dry_yc_5-dark.tdms",
                   "G:\PlantDataV2\dry\dry_yc_7-dark_0.tdms",
                   "G:\PlantDataV2\dry\dry_yc_12-dark.tdms",
                   "G:\PlantDataV2\dry\dry_yc_15-dark.tdms",
                   "G:\PlantDataV2\dry\dry_yc_18-dark.tdms"]"""

tdms_file_paths = ["G:\PlantDataV2\dry\dry_col2-dark.tdms",
                   "G:\PlantDataV2\dry\dry_col3-dark.tdms",
                   "G:\PlantDataV2\dry\dry_col4-dark.tdms",
                   "G:\PlantDataV2\dry\dry_fb3-dark.tdms",
                   "G:\PlantDataV2\dry\dry_gl_10-dark.tdms",
                   "G:\PlantDataV2\dry\dry_ws1-dark.tdms",
                   "G:\PlantDataV2\dry\dry_ws2-dark.tdms",
                   "G:\PlantDataV2\dry\dry_xhs_3-dark.tdms",
                   "G:\PlantDataV2\dry\dry_xhs_4-dark.tdms",
                   ]
channel_name = "声压_3 (滤波)"
sampling_rate = 204800
segment_length = 0.002
oaspl_threshold = 43.50
base_save_dir = "G:\\picSave\\plantV2"

main(tdms_file_paths, channel_name, sampling_rate, segment_length, oaspl_threshold, base_save_dir)

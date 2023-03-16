import numpy as np
import scipy as sp
import os
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D

path = '16mics_25gain_11th/'
# 读取音频文件
# fs, data1 = wavfile.read(os.path.join('../micstest/', path, '.aup-01.wav'))
# _, data2 = wavfile.read(os.path.join('../micstest/', path, '.aup-16.wav'))
# _, data3 = wavfile.read(os.path.join('../micstest/', path, '.aup-15.wav'))
# _, data4 = wavfile.read(os.path.join('../micstest/', path, '.aup-14.wav'))
# _, data5 = wavfile.read(os.path.join('../micstest/', path, '.aup-13.wav'))
# _, data6 = wavfile.read(os.path.join('../micstest/', path, '.aup-12.wav'))
# _, data7 = wavfile.read(os.path.join('../micstest/', path, '.aup-11.wav'))
# _, data8 = wavfile.read(os.path.join('../micstest/', path, '.aup-10.wav'))
# _, data9 = wavfile.read(os.path.join('../micstest/', path, '.aup-09.wav'))
# _, data10 = wavfile.read(os.path.join('../micstest/', path, '.aup-08.wav'))
# _, data11 = wavfile.read(os.path.join('../micstest/', path, '.aup-07.wav'))
# _, data12 = wavfile.read(os.path.join('../micstest/', path, '.aup-06.wav'))
# _, data13 = wavfile.read(os.path.join('../micstest/', path, '.aup-05.wav'))
# _, data14 = wavfile.read(os.path.join('../micstest/', path, '.aup-04.wav'))
# _, data15 = wavfile.read(os.path.join('../micstest/', path, '.aup-03.wav'))
# _, data16 = wavfile.read(os.path.join('../micstest/', path, '.aup-02.wav'))
# data = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16]
# data = (np.array(data)).transpose(1,0)   # 维度 ： (nsamples , nchannels)

fs, data = wavfile.read('../ov1_split1/wav_ov1_split1_30db/test_0_desc_30_100.wav')
eps = np.spacing(np.float(1e-16))
data = data / 32768.0 + eps   # 标准化
# 打印采样率和数据长度
print(f"采样率: {fs} Hz")
print("数据维度:", data.shape)

# 麦克风数量
num_mics = 8
# 声源数量
num_sources = 2
# 阵列半径
d = 0.05
# 麦克风位置
mic_locs = np.zeros((num_mics, 3))
for i in range(num_mics):
    mic_locs[i] = [d*np.cos(2*np.pi*i/num_mics), d*np.sin(2*np.pi*i/num_mics), 0]
# 声速
cc = 343
mic_locs = mic_locs.T  # 维度: (3, num_mics)

# 选择要读取的时间窗口
start_time = 2.201848609 # seconds
end_time = 3.29785767929  # seconds
start_index = int(start_time * fs)
end_index = int(end_time * fs)
data = data[start_index-1:end_index, :]

# 分辨率
resolution = 5
azi_list = range(-180, 180, resolution)
ele_list = range(-60, 60, resolution)

#stft
winn = 512
nfft = 512
hop = winn//2
f, t, Sxx = signal.stft(data[:, 0], fs, window='hann', nperseg=winn, noverlap=hop)
nbin = len(f)
nframe = len(t)
spectra = np.zeros((nbin, nframe, num_mics), dtype=complex)
for i in range(num_mics):
    f, t, Sxx = signal.stft(data[:, i], fs, window='hann', nperseg=winn, noverlap=hop)
    spectra[:, :, i] = Sxx  # nbin x nfram x nchan matrix
spectra = spectra[1:,:,:]

# MUSIC算法
nbins = spectra.shape[0]  # nbins
Rxx = np.zeros((num_mics, num_mics), dtype='complex')  # 协方差矩阵  维度： （nchannels，nchannels）
p_spectrum = np.zeros(((nbins),len(azi_list), len(ele_list)))  # 定义空间谱图
freq  = np.arange(1, nfft//2+1) * fs/nfft   # stft变换的频率点

for ibin in range(nbins):
    X = spectra[ibin, :, :]     # X: (nframes.nchannels)
    Rxx = X.T @ np.conj(X)  # 计算自相关矩阵，得到形状为 (nchan, nchan) 的二维数组
    print(f"对于每个频点: {ibin}")

    # 计算特征值和特征向量
    lamb, E = np.linalg.eig(Rxx)
    # 取噪声子空间
    idx = np.argsort(lamb)[::-1]  # 从大到小排序
    E = E[:, idx]
    En = E[:, num_sources:]

    for i, azi in enumerate(azi_list):
        for j, ele in enumerate(ele_list):
            v =np.array([[np.cos(np.radians(ele))*np.cos(np.radians(azi))],
                         [np.cos(np.radians(ele))*np.sin(np.radians(azi))],
                         [np.sin(np.radians(ele))]])
            tau = np.dot(v.T, mic_locs)
            a = np.exp(1j * 2 * np.pi * freq[ibin] / cc * tau.T)   # 阵列响应向量
            p_spectrum[ibin, i, j] = 1 / np.sum(np.abs(np.dot(np.dot(np.conj(a).T, En), np.dot(np.conj(En).T, a))))   # 空间谱图
# 可视化
spec = np.sum(p_spectrum, axis=0)  # nAzi x nEle
spec /= np.max(spec)

# 声源定位azi角度
spec_azi = np.sum(spec,axis=1)/len(ele_list)  # nAzi
plt.figure()
plt.plot(spec_azi)
plt.xlabel('row')
plt.ylabel('Value')
plt.title('spec_azi')
max_idx = np.argmax(spec_azi)
print("声源定位azi角度：", (azi_list[0]+max_idx*resolution))

# MUSIC空间谱图
plt.figure()
ax = plt.axes(projection='3d')
xx = np.arange(-180,180,resolution)
yy = np.arange(-60,60,resolution)
X,Y = np.meshgrid(xx,yy)
ax.plot_surface(X, Y, spec.T, cmap = "rainbow")
plt.show()
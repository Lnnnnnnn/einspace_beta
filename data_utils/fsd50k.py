import os
import tqdm
import glob
import numpy as np
import librosa
import torch
import torchaudio
import json
import random
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
from scipy import stats
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader

'''
Metrics
'''
def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            # y_true, y_pred
            target[:, k], output[:, k], average=None)
        # print("\ttarget", target[:, k])
        # print("\toutput", output[:, k])
        # print("\tAP", avg_precision)

        # AUC
        # auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        # (precisions, recalls, thresholds) = metrics.precision_recall_curve(
        #     target[:, k], output[:, k])

        # FPR, TPR
        # (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        # save_every_steps = 1000     # Sample statistics to reduce size
        dict = {
            # 'precisions': precisions[0::save_every_steps],
            # 'recalls': recalls[0::save_every_steps],
            'AP': avg_precision,
            # 'fpr': fpr[0::save_every_steps],
            # 'fnr': 1. - tpr[0::save_every_steps],
            # 'auc': auc
        }
        stats.append(dict)

    return stats


def calculate_map(test_gts, test_pred_batches):
    # [T, V, 200], # [T, 200]
    # print(len(test_gts), len(test_gts[0]), len(test_pred_batches), len(test_pred_batches[0]))
    test_gts = test_gts.squeeze(1)
    test_pred = torch.stack([torch.tensor(p).mean(0) for p in test_pred_batches])
    # test_pred = test_pred_batches.mean(1) # [T, 200]
    # print(test_gts.shape, test_pred.shape)
    test_predictions = torch.sigmoid(test_pred)
    test_predictions = np.asarray(test_predictions).astype('float32')
    test_gts = np.asarray(test_gts).astype('int32')
    # print("test pred batches", test_pred_batches[0])
    # print("test pred", test_pred[0])
    # print("test predictions", test_predictions[0])
    # print("test gts", test_gts[0])
    # print("preds > 0.5", test_predictions[0][test_predictions[0] > 0.5])
    # print("gts", np.nonzero(test_gts[0]))

    stats = calculate_stats(test_predictions, test_gts)
    # print(stats)
    mAP = np.mean([stat['AP'] for stat in stats])
    # mAUC = np.mean([stat['auc'] for stat in stats])
    return mAP

'''
Audio Mixer
'''
def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image, _, rnd_target = dataset.__get_item_helper__(rnd_idx)
    return rnd_image, rnd_target


class BackgroundAddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_idx = random.randint(0, dataset.get_bg_len() - 1)
        rnd_image = dataset.get_bg_feature(rnd_idx)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        return image, target


class AddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        target = (1 - alpha) * target + alpha * rnd_target
        target = torch.clip(target, 0.0, 1.0)
        return image, target


class SigmoidConcatMixer:
    def __init__(self, sigmoid_range=(3, 12)):
        self.sigmoid_range = sigmoid_range

    def sample_mask(self, size):
        x_radius = random.randint(*self.sigmoid_range)

        step = (x_radius * 2) / size[1]
        x = torch.arange(-x_radius, x_radius, step=step).float()
        y = torch.sigmoid(x)
        mix_mask = y.repeat(size[0], 1)
        return mix_mask

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        mix_mask = self.sample_mask(image.shape[-2:])
        rnd_mix_mask = 1 - mix_mask

        image = mix_mask * image + rnd_mix_mask * rnd_image
        target = target + rnd_target
        target = torch.clip(target, 0.0, 1.0)
        return image, target


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target = mixer(dataset, image, target)
        return image, target


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target

'''
Dataset Transforms
'''
def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


# Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec


class SpecAugment:
    def __init__(self,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
                            self.num_mask,
                            self.freq_masking,
                            self.time_masking,
                            image.min())


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class ToTensor:
    def __call__(self, array):
        return torch.from_numpy(array).float()


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[1] - self.size)
        return signal[:, start: start + self.size]


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):

        if signal.shape[1] > self.size:
            start = (signal.shape[1] - self.size) // 2
            return signal[:, start: start + self.size]
        else:
            return signal


class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width,
                                'constant', constant_values=signal.min())
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal


def get_transforms_fsd_chunks(train, size,
                              wrap_pad_prob=0.5,
                              spec_num_mask=2,
                              spec_freq_masking=0.15,
                              spec_time_masking=0.20,
                              spec_prob=0.5):
    if train:
        transforms = Compose([
            OneOf([
                PadToSize(size, mode='wrap'),
                PadToSize(size, mode='constant'),
            ], p=[wrap_pad_prob, 1 - wrap_pad_prob]),
            UseWithProb(SpecAugment(num_mask=spec_num_mask,
                                    freq_masking=spec_freq_masking,
                                    time_masking=spec_time_masking), spec_prob),
            RandomCrop(size),       # it's okay, our chunks are of exact `size` timesteps anyway
            ToTensor()
        ])
    else:
        transforms = Compose([
            PadToSize(size),
            # CenterCrop(size),
            ToTensor()
        ])
    return transforms



'''
Dataloader Collate Functions
'''
def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
        targets.append(target.unsqueeze(0))
    targets = torch.cat(targets)
    return inputs, inputs_complex, targets


def _collate_fn_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
    targets = torch.LongTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
        targets[x] = target
    return inputs, inputs_complex, targets

'''
Audio Parsing
'''
def load_audio(f, sr, min_duration: float = 5.):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    x, clip_sr = sf.read(f)
    x = x.astype('float32')
    assert clip_sr == sr

    # min filtering and padding if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x

class AudioParser(object):
    def __init__(self, n_fft=511, win_length=None, hop_length=None, sample_rate=22050,
                 feature="spectrogram", top_db=150):
        super(AudioParser, self).__init__()
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.n_fft//2 if hop_length is None else hop_length
        assert feature in ['melspectrogram', 'spectrogram']
        self.feature = feature
        self.top_db = top_db
        if feature == "melspectrogram":
            self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=96 * 20,
                                                                win_length=int(sample_rate * 0.03),
                                                                hop_length=int(sample_rate * 0.01),
                                                                n_mels=96)
        else:
            self.melspec = None

    def __call__(self, audio):
        if self.feature == 'spectrogram':
            # TOP_DB = 150
            comp = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length)
            real = np.abs(comp)
            real = librosa.amplitude_to_db(real, top_db=self.top_db)
            real += self.top_db / 2

            mean = real.mean()
            real -= mean        # per sample Zero Centering
            return real, comp

        elif self.feature == 'melspectrogram':
            # melspectrogram features, as per FSD50k paper
            x = torch.from_numpy(audio).unsqueeze(0)
            specgram = self.melspec(x)[0].numpy()
            specgram = librosa.power_to_db(specgram)
            specgram = specgram.astype('float32')
            specgram += self.top_db / 2
            mean = specgram.mean()
            specgram -= mean
            return specgram, specgram

'''
Dataset Classes
'''

class SpectrogramDataset(Dataset):
    def __init__(self, manifest_path: str, labels_map: str,
                 audio_config: dict, mode: Optional[str] = "multilabel",
                 augment: Optional[bool] = False,
                 labels_delimiter: Optional[str] = ",",
                 mixer: Optional = None,
                 transform: Optional = None) -> None:
        super(SpectrogramDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)

        self.len = None
        self.labels_delim = labels_delimiter
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values
        self.labels = df['labels'].values
        self.exts = df['ext'].values
        self.unique_exts = np.unique(self.exts)
        assert len(self.files) == len(self.labels) == len(self.exts)
        self.len = len(self.unique_exts)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", True)
        self.augment = augment
        self.min_duration = audio_config.get("min_duration", None)
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        else:
            self.bg_files = None

        self.mode = mode
        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, feature=feature)
        self.mixer = mixer
        self.transform = transform

        if self.bg_files is not None:
            print("prepping bg_features")
            self.bg_features = []
            for f in tqdm.tqdm(self.bg_files):
                preprocessed_audio = self.__get_audio__(f)
                real, comp = self.__get_feature__(preprocessed_audio)
                self.bg_features.append(real)
        else:
            self.bg_features = None
        self.prefetched_labels = None
        if self.mode == "multilabel":
            self.fetch_labels()

    def fetch_labels(self):
        self.prefetched_labels = []
        for lbl in tqdm.tqdm(self.labels):
            proc_lbl = self.__parse_labels__(lbl)
            self.prefetched_labels.append(proc_lbl.unsqueeze(0))
        self.prefetched_labels = torch.cat(self.prefetched_labels, dim=0)

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def get_bg_feature(self, index: int) -> torch.Tensor:
        if self.bg_files is None:
            return None
        real = self.bg_features[index]
        if self.transform is not None:
            real = self.transform(real)
        return real

    def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        lbls = self.labels[index]
        label_tensor = self.__parse_labels__(lbls)
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_ext = self.unique_exts[index]
        idxs = np.where(self.exts == tgt_ext)[0]
        rand_index = np.random.choice(idxs)
        
        real, comp, label_tensor = self.__get_item_helper__(rand_index)
        if self.mixer is not None:
            real, final_label = self.mixer(self, real, label_tensor)
            if self.mode != "multiclass":
                # reshape from (96, 101) to (1, 101, 96)
                real = real.permute(1, 0).unsqueeze(0)
                return real, final_label

        # reshape from (96, 101) to (1, 101, 96)
        real = real.permute(1, 0).unsqueeze(0)

        return real, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            return self.labels_map[lbls]

    def __len__(self):
        return self.len

    def get_bg_len(self):
        return len(self.bg_files)

'''
Audio Eval 
'''
class FSD50kEvalDataset(Dataset):
    def __init__(self, manifest_path: str, labels_map: str,
                 audio_config: dict,
                 labels_delimiter: Optional[str] = ",",
                 transform: Optional = None) -> None:
        super(FSD50kEvalDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)

        self.len = None
        self.labels_delim = labels_delimiter
        df = pd.read_csv(manifest_path)
        self.files = df['files'].values
        self.labels = df['labels'].values
        self.exts = df['ext'].values
        self.unique_exts = np.unique(self.exts)

        assert len(self.files) == len(self.labels) == len(self.exts)
        self.len = len(self.unique_exts)
        # print("LENGTH OF VAL SET:", self.len)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", False)
        self.min_duration = audio_config.get("min_duration", None)

        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                       hop_length=self.hop_len, feature=feature)
        self.transform = transform

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        lbls = self.labels[index]
        label_tensor = self.__parse_labels__(lbls)
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.transform is not None:
            real = self.transform(real)
        return real, comp, label_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_ext = self.unique_exts[index]
        idxs = np.where(self.exts == tgt_ext)[0]

        tensors = []
        label_tensors = []
        for idx in idxs:
            real, comp, label_tensor = self.__get_item_helper__(idx)
            tensors.append(real.unsqueeze(0).unsqueeze(0))
            label_tensors.append(label_tensor.unsqueeze(0))

        tensors = torch.cat(tensors)
        tensors = tensors.permute(0, 1, 3, 2)
        return tensors, label_tensors[0]

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        for lbl in lbls.split(self.labels_delim):
            label_tensor[self.labels_map[lbl]] = 1

        return label_tensor

    def __len__(self):
        return self.len


def _collate_fn_eval(batch):
    return batch[0][0], batch[0][1]

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def reverseTransform(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if len(img.shape) == 5:
        for i in range(3):
            img[:, i, :, :, :] = img[:, i, :, :, :]*std[i] + mean[i]
    else:
        for i in range(3):
            img[:, i, :, :] = img[:, i, :, :]*std[i] + mean[i]
    return img
    

class BackgroundAddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_idx = random.randint(0, dataset.get_bg_len() - 1)
        rnd_image = dataset.get_bg_feature(rnd_idx)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        return image, target

class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target


def build_nasbench360_fsd_dataset(split, cfg_dict):
    assert split in ["train", "val", "trainval", "test"]
    root_path = cfg_dict["fsd_root"]
    audio_cfg = dict(
        feature="melspectrogram",
        sample_rate=22050,
        min_duration=1,
        bg_files=os.path.join(root_path, "noise_22050"),
    )

    if split == "train":
        if not os.path.exists(os.path.join('/localdisk/home/lericsso/code/einspace/data/train_dataset.pt')):
            mixer = BackgroundAddMixer()
            tr_mixer = UseMixerWithProb(mixer, 0.75)
            tr_tfs = get_transforms_fsd_chunks(True, 101)
            train_dataset = SpectrogramDataset(
                os.path.join(root_path, "tr.csv"),
                os.path.join(root_path, "lbl_map.json"),
                audio_cfg, 
                mode="multilabel", 
                augment=True,
                mixer=tr_mixer,
                transform=tr_tfs,
            )
            # Save the train_dataset
            torch.save(train_dataset, '/localdisk/home/lericsso/code/einspace/data/train_dataset.pt')
        else:
            train_dataset = torch.load('/localdisk/home/lericsso/code/einspace/data/train_dataset.pt')
        return train_dataset
    elif split == "val":
        if not os.path.exists(os.path.join('/localdisk/home/lericsso/code/einspace/data/val_dataset.pt')):
            val_tfs = get_transforms_fsd_chunks(False, 101)
            val_dataset = FSD50kEvalDataset(
                os.path.join(root_path, "val.csv"), 
                os.path.join(root_path, "lbl_map.json"),
                audio_cfg,
                transform=val_tfs,
            )
            # Save the val_dataset
            torch.save(val_dataset, '/localdisk/home/lericsso/code/einspace/data/val_dataset.pt')
        else:
            val_dataset = torch.load('/localdisk/home/lericsso/code/einspace/data/val_dataset.pt')
        return val_dataset
    elif split == "trainval":
        if not os.path.exists(os.path.join('/localdisk/home/lericsso/code/einspace/data/trainval_dataset.pt')):
            mixer = BackgroundAddMixer()
            trval_mixer = UseMixerWithProb(mixer, 0.75)
            trval_tfs = get_transforms_fsd_chunks(True, 101)
            trainval_dataset = SpectrogramDataset(
                os.path.join(root_path, "trval.csv"),
                os.path.join(root_path, "lbl_map.json"),
                audio_cfg, 
                mode="multilabel", 
                augment=True,
                mixer=trval_mixer,
                transform=trval_tfs,
            )
            # Save the val_dataset
            torch.save(trainval_dataset, '/localdisk/home/lericsso/code/einspace/data/trainval_dataset.pt')
        else:
            trainval_dataset = torch.load('/localdisk/home/lericsso/code/einspace/data/trainval_dataset.pt')
        return trainval_dataset
    elif split == "test":
        if not os.path.exists(os.path.join('/localdisk/home/lericsso/code/einspace/data/test_dataset.pt')):
            test_tfs = get_transforms_fsd_chunks(False, 101)
            test_dataset = FSD50kEvalDataset(
                os.path.join(root_path, "eval.csv"), 
                os.path.join(root_path, "lbl_map.json"),
                audio_cfg,
                transform=test_tfs,
            )
            # Save the test_dataset
            torch.save(test_dataset, '/localdisk/home/lericsso/code/einspace/data/test_dataset.pt')
        else:
            test_dataset = torch.load('/localdisk/home/lericsso/code/einspace/data/test_dataset.pt')
        return test_dataset
    else:
        raise NotImplementedError


if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    #                                 Training
    # ----------------------------------------------------------------------------
    audio_cfg = dict(
        feature="melspectrogram",
        sample_rate=22050,
        min_duration=1,
        bg_files="/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/noise_22050"
    )

    # mixer = BackgroundAddMixer()
    # tr_mixer = UseMixerWithProb(mixer, 0.75)
    # tr_tfs = get_transforms_fsd_chunks(True, 101)

    # train_dataset = SpectrogramDataset(
    #     "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/tr.csv",
    #     "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/lbl_map.json",
    #     audio_cfg, 
    #     mode="multilabel", 
    #     augment=True,
    #     mixer=tr_mixer,
    #     transform=tr_tfs
    # )
    # print(train_dataset)
    # print(train_dataset[0][0].shape, train_dataset[0][1].shape)  # data, label

    # ----------------------------------------------------------------------------
    #                                 Validation
    # ----------------------------------------------------------------------------

    # val_tfs = get_transforms_fsd_chunks(False, 101)

    # val_dataset = FSD50kEvalDataset(
    #     "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/val.csv", 
    #     "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/lbl_map.json",
    #     audio_cfg,
    #     transform=val_tfs
    # )
    # print(val_dataset)
    # # print(val_dataset[0][0].shape, val_dataset[0][1].shape)  # data, label
    # print(val_dataset[0])


    # ----------------------------------------------------------------------------
    #                                 Testing
    # ----------------------------------------------------------------------------
    test_tfs = get_transforms_fsd_chunks(False, 101)

    test_dataset = FSD50kEvalDataset(
        "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/eval.csv", 
        "/localdisk/home/lericsso/code/einspace/data/fsd50k/nasbench360/chunks_meta/lbl_map.json",
        audio_cfg,
        transform=test_tfs
    )
    print(test_dataset)
    print(test_dataset[0][1].shape)

    # compute evaluation metric
    k = 50
    pseudo_pred = torch.rand((k, 200)).numpy().astype('float32')
    gts = [test_dataset[i][1].numpy()[0] for i in range(k)]
    gts = np.asarray(gts).astype('int32')
    print(pseudo_pred.shape, gts.shape)
    stats = calculate_stats(pseudo_pred, gts)
    mAP = np.mean([stat['AP'] for stat in stats])
    print("1 - mAP: {:.6f}".format(1 - mAP))

import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import logging as logger
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
from sklearn.metrics import f1_score

from losses import MyInfoNCE
from models.vit import FOCAL_ViT
from models.hrnet import FOCAL_HRNet

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='train', help='one of [train, val, test_single, flist]')
parser.add_argument('--input_size', type=int, default=1024, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=16, help='resolution of input / output, 4 for HRNet and 16 for ViT')
parser.add_argument('--train_bs', type=int, default=4, help='training batch size')
parser.add_argument('--test_bs', type=int, default=8, help='testing batch size')
parser.add_argument('--save_res', type=int, default=1, help='whether to save the output')
parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPU ID')
parser.add_argument('--metric', type=str, default='cosine', help='metric for loss and clustering')
parser.add_argument('--path_input', type=str, default='demo/input/', help='path of input forgeries')
parser.add_argument('--path_gt', type=str, default='demo/gt/', help='path of ground-truth (could be empty)')
parser.add_argument('--nickname', type=str, default='demo', help='short name for the dataset')
args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, num=0, file='', choice='train'):
        self.num = num
        self.choice = choice
        self.filelist = file

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.size = args.input_size
        self.albu = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),
            A.PadIfNeeded(min_height=self.size, min_width=self.size, p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
            ], p=0.75),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
            A.OneOf([
                A.OneOf([
                    A.Blur(p=1),
                    A.GaussianBlur(p=1),
                    A.MedianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=1),
                A.OneOf([
                    A.Downscale(p=1),
                    A.GaussNoise(p=1),
                    A.ISONoise(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.RandomToneCurve(p=1),
                    A.Sharpen(p=1),
                ], p=1),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
                    A.GridDistortion(p=1),
                ], p=1),
            ], p=0.25),
        ])

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        fname1, fname2 = self.filelist[idx]

        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        mask = cv2.imread(fname2) if fname2 != '' else np.zeros([H, W, 3])
        mask = thresholding(mask)

        if self.choice == 'train' and random.random() < 0.75:
            aug = self.albu(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size // args.gt_ratio, self.size // args.gt_ratio))
        mask = thresholding(mask)

        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), H, W, fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class FOCAL(nn.Module):
    def __init__(self, net_list=[('ViT', '')]):
        super(FOCAL, self).__init__()
        self.lr = 1e-4
        self.network_list = []
        for net_name, net_weight in net_list:
            if net_name == 'HRNet':
                cur_net = FOCAL_HRNet()
            elif net_name == 'ViT':
                cur_net = FOCAL_ViT()
            else:
                logger.info('Error: Undefined Network.')
                exit()
            cur_net = nn.DataParallel(cur_net).cuda()
            if net_weight != '':
                self.load(cur_net, net_weight)
            self.network_list.append(cur_net)

        self.extractor_optimizer = optim.Adam(self.network_list[0].parameters(), lr=self.lr)
        # self.extractor_optimizer = optim.AdamW(self.network_list[0].parameters(), lr=self.lr)
        self.save_dir = 'weights/' + args.out_dir
        if args.type == 'train':
            rm_and_make_dir(self.save_dir)
        self.myInfoNCE = MyInfoNCE(metric=args.metric)
        self.clustering = KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)

    def process(self, Ii, Mg, isTrain=False):
        self.extractor_optimizer.zero_grad()

        if isTrain:
            Fo = self.network_list[0](Ii)
            Fo = Fo.permute(0, 2, 3, 1)
            B, H, W, C = Fo.shape
            Fo = F.normalize(Fo, dim=3)
        else:
            with torch.no_grad():
                Fo = self.network_list[0](Ii)
                Fo = Fo.permute(0, 2, 3, 1)
                B, H, W, C = Fo.shape
                Fo = F.normalize(Fo, dim=3)
                Fo_list = [Fo]
                for additional_net in self.network_list[1:]:
                    Fo_add = additional_net(Ii)
                    Fo_add = F.interpolate(Fo_add, (H, W))
                    Fo_add = Fo_add.permute(0, 2, 3, 1)
                    Fo_add = F.normalize(Fo_add, dim=3)
                    Fo_list.append(Fo_add)
                Fo = torch.cat(Fo_list, dim=3)

        if isTrain:
            info_nce_loss = []
            for idx in range(B):
                Fo_idx = Fo[idx]
                Mg_idx = Mg[idx][0]
                query = Fo_idx[Mg_idx == 0]
                negative = Fo_idx[Mg_idx == 1]
                if negative.size(0) == 0 or query.size(0) == 0:
                    continue
                dict_size = 1000  # could be larger according to gpu memory
                query_sample = query[torch.randperm(query.size()[0])[:dict_size]]
                negative_sample = negative[torch.randperm(negative.size(0))[:dict_size]]
                info_nce_loss.append(self.myInfoNCE(query_sample, query_sample, negative_sample))

            batch_loss = torch.mean(torch.stack(info_nce_loss).squeeze())
            self.backward(batch_loss)
            return batch_loss
        else:
            Mo = None
            Fo = torch.flatten(Fo, start_dim=1, end_dim=2)
            result = self.clustering(x=Fo, k=2)
            Lo_batch = result.labels
            for idx in range(B):
                Lo = Lo_batch[idx]
                if torch.sum(Lo) > torch.sum(1 - Lo):
                    Lo = 1 - Lo
                Lo = Lo.view(H, W)[None, :, :, None]
                Mo = torch.cat([Mo, Lo], dim=0) if Mo is not None else Lo
            Mo = Mo.permute(0, 3, 1, 2)
            return Mo

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward(retain_graph=False)
            self.extractor_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.network_list[0].state_dict(),
                   self.save_dir + path + '%s_weights.pth' % self.network_list[0].module.name)

    def load(self, extractor, path=''):
        weights_file = torch.load('weights/' + path)
        cur_weights = extractor.state_dict()
        for key in weights_file:
            if key in cur_weights.keys() and weights_file[key].shape == cur_weights[key].shape:
                cur_weights[key] = weights_file[key]
        extractor.load_state_dict(cur_weights)
        logger.info('Loaded [%s] from [%s]' % (extractor.module.name, path))


class ForgeryForensics():
    def __init__(self):
        self.train_npy_list = [
            # name, repeat_time
            ('tampCOCO_sp_199999.npy', 1),
            # ('tampCOCO_cm_199429.npy', 1),
            # ('tampCOCO_bcm_199443.npy', 1),
            # ('train_casia2_5123.npy', 40),
            # ('train_imd2020_2010.npy', 20),
        ]
        self.train_file = None
        for item in self.train_npy_list:
            self.train_file_tmp = np.load('flist/' + item[0])
            for _ in range(item[1]):
                self.train_file = np.concatenate(
                    [self.train_file, self.train_file_tmp]) if self.train_file is not None else self.train_file_tmp

        self.train_num = len(self.train_file)
        train_dataset = MyDataset(num=self.train_num, file=self.train_file, choice='train')

        self.val_npy_list = [
            # name, nickname

            # Validation Dataset:
            ('val_1000.npy', 'valid'),

            # Testing Dataset:
            # ('test_Coverage_100.npy', 'Cove'),
            # ('test_Columbia_160.npy', 'Colu'),
            # ('test_NIST16_564.npy', 'NIST'),
            # ('test_CASIA_920.npy', 'CASI'),
            # ('test_MultiSP_227.npy', 'MISD'),
            # ('test_FF++_1000.npy', 'FF++'),
        ]
        self.val_file_list = []
        for item in self.val_npy_list:
            self.val_file_tmp = np.load('flist/' + item[0])
            self.val_file_list.append(self.val_file_tmp)

        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        self.focal = FOCAL([
            ('ViT', ''),  # Training from scratch
            # ('ViT', 'FOCAL_ViT_weights.pth'),  # Fine-tune ViT
            # ('HRNet', 'FOCAL_HRNet_weights.pth'),  # Fine-tune HRNet
        ]).cuda()
        self.n_epochs = 99999
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, num_workers=self.train_bs,
                                       shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Test on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        cnt, batch_losses = 0, []
        best_score = 0
        scheduler = ReduceLROnPlateau(self.focal.extractor_optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-8)
        # scheduler = CosineAnnealingWarmRestarts(self.focal.gen_optimizer, T_0=2, T_mult=1)
        self.focal.train()
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                cnt += self.train_bs
                Ii, Mg = (item.cuda() for item in items[:2])  # Input, Ground-truth Mask
                batch_loss = self.focal.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())
                if cnt % (self.train_bs * 20) == 0:
                    logger.info('Tra (%6d/%6d): G:%5.4f' % (cnt, self.train_num, np.mean(batch_losses)))
                if cnt % int((self.train_loader.dataset.__len__() / 80) // self.train_bs * self.train_bs) == 0:
                    self.focal.save('latest/')
                    logger.info('Ep%03d(%6d/%6d): Tra: G:%5.4f' % (epoch, cnt, self.train_num, np.mean(batch_losses)))
                    tmp_score = self.val()
                    scheduler.step(tmp_score)
                    if tmp_score > best_score:
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best)' % best_score)
                        self.focal.save('Ep%03d_%5.4f/' % (epoch, tmp_score))
                    else:
                        logger.info('Score: %5.4f' % tmp_score)
                    self.focal.train()
                    batch_losses = []
            cnt = 0

    def val(self):
        tmp_score = []
        for idx in range(len(self.val_file_list)):
            P_F1, P_IOU = ForensicTesting(self.focal, bs=self.test_bs, test_npy=self.val_npy_list[idx][0],
                                          test_file=self.val_file_list[idx])
            tmp_score.append(P_F1)
            tmp_score.append(P_IOU)
            logger.info('%s(#%d): PF1:%5.4f, PIOU:%5.4f' % (
            self.val_npy_list[idx][1], len(self.val_file_list[idx]), P_F1, P_IOU))
        if args.type == 'val':
            logger.info('Score: %5.4f' % np.mean(tmp_score))
        return np.mean(tmp_score)


def ForensicTesting(model, bs=1, test_npy='', test_file=None):
    if test_file is None:
        test_file = np.load('flist/' + test_npy)
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, bs), shuffle=False)
    # model.eval()  # Fix Bug (Using following 'eval()')
    for net in model.network_list:
      net.eval()

    f1, iou = [], []
    if args.save_res == 1:
        path_out = 'demo/output/'
        rm_and_make_dir(path_out)

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.cuda() for item in items[:-1])
        filename = items[-1]

        Mo = model.process(Ii, None, isTrain=False)

        Mg, Mo = convert(Mg), convert(Mo)

        if args.save_res == 1:
            Hg, Wg = Hg.cpu().numpy(), Wg.cpu().numpy()
            for i in range(Ii.shape[0]):
                res = cv2.resize(Mo[i], (Wg[i].item(), Hg[i].item()))
                res = thresholding(res)
                cv2.imwrite(path_out + filename[i][:-4] + '.png', res.astype(np.uint8))

        for i in range(Mo.shape[0]):
            Mo_resized = thresholding(cv2.resize(Mo[i], (Mg[i].shape[:2][::-1])))[..., None]
            f1.append(f1_score(Mg[i].flatten(), Mo_resized.flatten(), average='macro'))
            iou.append(metric_iou(Mo_resized / 255., Mg[i] / 255.))

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    if args.type == 'test_single':
        logger.info('Score: F1: %5.4f, IoU: %5.4f' % (Pixel_F1, Pixel_IOU))
    return Pixel_F1, Pixel_IOU


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def generate_flist(path_input, path_gt, nickname):
    # NOTE: The image and ground-truth should have the same name.
    # Example:
    # path_input = 'tampCOCO/sp_images/'
    # path_gt = 'tampCOCO/sp_masks/'
    # nickname = 'tampCOCO_sp'
    res = []
    flag = False
    flist = sorted(os.listdir(path_input))
    for file in flist:
        name = file.rsplit('.', 1)[0]
        path_mask = path_gt + name + '.png'
        # path_mask = path_gt + name + '.tif'
        # path_mask = path_gt + name + '_gt.png'
        if not os.path.exists(path_mask):
            path_mask = ''
            flag = True
        res.append((path_input + file, path_mask))
    save_name = '%s_%s.npy' % (nickname, len(res))
    np.save('flist/' + save_name, np.array(res))
    if flag:
        logger.info('Note: The following score is meaningless since no ground-truth is provided.')
    return save_name


def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou


if __name__ == '__main__':
    if args.type == 'train':
        model = ForgeryForensics()
        model.train()
    elif args.type == 'val':
        model = ForgeryForensics()
        model.val()
    elif args.type == 'test_single':
        model = FOCAL(net_list=[
            ('ViT', 'FOCAL_ViT_weights.pth'),
            ('HRNet', 'FOCAL_HRNet_weights.pth'),
        ]).cuda()
        file_npy = generate_flist(args.path_input, args.path_gt, args.nickname)
        ForensicTesting(model, test_npy=file_npy)
    elif args.type == 'flist':
        generate_flist(args.path_input, args.path_gt, args.nickname)

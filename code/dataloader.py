import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
class S5P_P25(Dataset):

    def __init__(self, path, device='cpu', transform=False):
        self.path = path
        self.device = device
        self.transform = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]) if transform else None

    def __len__(self):

        return len(self.path)

    def __getitem__(self, index):

        patch = torch.load(self.path[index])
        if not self.transform == None:
            patch = self.transform(patch)

        cams = patch[-1:, :, :]/ 63
        s5p = patch[:-1, :, :]/ 2e-6

        # B, H, W = s5p.shape
        # rH = H % 8
        # rW = W % 8
        # minr = rH // 2
        # minw = rW // 2
        # maxr = H if rH == 0 else -rH // 2
        # maxw = W if rW == 0 else -rW // 2
        # s5p = s5p[:, minr:maxr, minw:maxw]
        # cams = cams[:, minr:maxr, minw:maxw]

        return s5p.to(self.device), cams.to(self.device)
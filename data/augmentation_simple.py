import numpy as np
from scipy.linalg import expm, norm
import cv2
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
cv2.setNumThreads(1)


class TrainTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [RandomRotation(axis=np.array([0, 1, 0])), RandomCenterCrop(radius=76)]
        elif self.aug_mode == 1:
            t = [RandomRotation(max_theta=0,axis=np.array([0, 1, 0])), 
                RandomCenterCrop(radius=76),
                JitterPoints(sigma=1,clip=2),
                RemoveRandomPoints(r=(0.01, 0.1))]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))       
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

class ValTransform:
    def __init__(self):
        self.aug_mode = 0
        t = [RandomRotation(axis=np.array([0, 1, 0])), RandomCenterCrop(radius=76)]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

class TrainRGBTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        elif self.aug_mode == 1:
            t = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 transforms.RandomErasing(p=0.75, scale=(0.01, 0.10), ratio=(0.2, 0.8), value='random', inplace=True),
                 AddGaussianNoise(mean=0.0, std=0.01)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.aug_mode == 0:
            e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
            e = Image.fromarray(e)
            e = self.transform(e)
        else:
            e = rotate_panorama(e, max_delta=5)
            e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
            e = Image.fromarray(e)
            e = self.transform(e)
        return e

class ValRGBTransform:
    def __init__(self):
        self.aug_mode = 0
        t = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
        e = Image.fromarray(e)
        e = self.transform(e)
        return e

class TrainTileTransform:
    def __init__(self, aug_mode, tile_size=224):
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            t = [transforms.Resize(size=(tile_size,tile_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        elif self.aug_mode == 1:
            t = [transforms.Resize(size=(tile_size,tile_size)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 transforms.RandomErasing(p=0.75, scale=(0.01, 0.10), ratio=(0.2, 0.8), value='random', inplace=True),
                 AddGaussianNoise(mean=0.0, std=0.01)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

class ValTileTransform:
    def __init__(self, tile_size=224):
        self.aug_mode = 0
        t = [transforms.Resize(size=(tile_size,tile_size)),
             transforms.ToTensor(), 
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

# point cloud
class RandomRotation:
    def __init__(self, max_theta=0, axis=np.array([0, 1, 0])):
        self.axis = axis
        self.max_theta = max_theta # Rotation around axis

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, e):
        rot = np.random.uniform(-self.max_theta, self.max_theta) 
        center = torch.tensor(e['center'])
        coords = e['cloud']
        R = self._M(self.axis, (2*np.pi - np.pi * (e['heading'] + rot) / 180))        
        coords = ((coords.sub(center))@R).add(center)
        e['cloud'] = coords
        return e


class RandomTranslation:
    def __init__(self, max_delta=0):
        self.max_delta = max_delta

    def __call__(self, e):
        trans = np.random.uniform(-self.max_delta, self.max_delta, size=(3))
        trans = torch.tensor([trans[0],0,trans[2]])
        coords = e['cloud'] + trans
        e['cloud'] = coords
        return e


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        coords = e['cloud']
        sample_shape = (coords.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(coords[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        coords[mask] = coords[mask] + jitter
        e['cloud'] = coords
        return e


class RandomCenterScale:
    def __init__(self, scale=1, rnd=0):
        self.scale = scale
        self.rnd = rnd
    
    def __call__(self, e):
        center = torch.tensor(e['center'])
        s = self.scale + np.random.uniform(-self.rnd, self.rnd)
        coords = e['cloud']
        coords = (coords.sub(center))*s.add(center)
        e['cloud'] = coords
        return e


# points drop out
class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        coords = e['cloud']
        feats = e['cloud_ft']
        n = len(coords)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        saved_mask = np.setdiff1d(np.arange(n), mask)
        coords = torch.index_select(coords, dim=0, index = torch.tensor(saved_mask))
        feats = torch.index_select(feats, dim=0, index = torch.tensor(saved_mask))
        e['cloud'] = coords
        e['cloud_ft'] = feats
        return e


class RandomCenterCrop:
    def __init__(self, radius=60, rnd=0):
        self.radius = radius
        self.rnd = rnd

    def __call__(self, e):
        r = self.radius - np.random.uniform(-self.rnd, self.rnd)
        center = e['center']
        coords = e['cloud']
        feats = e['cloud_ft']
        min_x = center[0] - r
        min_y = center[2] - r
        max_x = center[0] + r
        max_y = center[2] + r
        saved_mask = (min_x < coords[..., 0]) & (coords[..., 0] < max_x) & (min_y < coords[..., 2]) & (coords[..., 2] < max_y)
        coords = coords[saved_mask]
        feats = feats[saved_mask]
        e['cloud'] = coords
        e['cloud_ft'] = feats
        return e


class PointcloudSample:
    def __init__(self, npoints=8192):
        self.npoints = npoints

    def __call__(self, e):
        coords = e['cloud'].numpy()
        feats = e['cloud_ft'].numpy()
        pt_idxs = np.arange(0, coords.shape[0])
        np.random.shuffle(pt_idxs)
        if len(coords) > self.npoints:
            coords = coords[pt_idxs[0:self.npoints]]
            feats = feats[pt_idxs[0:self.npoints]]
        else:
            pt_idxs = np.random.choice(pt_idxs, size=self.npoints, replace=True)
            coords = coords[pt_idxs]
            feats = feats[pt_idxs]     
        e['cloud'] = torch.tensor(coords, dtype=torch.float)
        e['cloud_ft'] = torch.tensor(feats, dtype=torch.float)
        return e


# image
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def rotate_panorama(image, roll=0.0, pitch=0.0, yaw=0.0, max_delta=0):
    """ Rotates a equirectangular image given angles

        Parameters:
            image (Numpy array) -> equirectangular image
            roll (float) -> Roll angle in degrees
            pitch (float) -> Pitch angle in degrees 
            yaw (float) -> Yaw angle in degrees 
    """
    yaw = yaw + np.random.uniform(-max_delta, max_delta)
    h,w = image.shape[0:2]
    euler_angles = np.radians(np.array([roll,pitch,yaw]))
    [R, _] = cv2.Rodrigues(euler_angles) 
    
    # Project equirectangular points to original sphere
    lat = np.pi / h * np.arange(0,h)
    lat = np.expand_dims(lat,1)
    lon = (2*np.pi / w * np.arange(0,w)).T
    lon = np.expand_dims(lon,0)

    # Convert to cartesian coordinates
    x = np.sin(lat)*np.cos(lon)
    y = np.sin(lat)*np.sin(lon)
    z = np.tile(np.cos(lat), [1,w])

    # Rotate points
    xyz = np.stack([x,y,z],axis=2).reshape(h*w,3).T
    rotated_points = np.dot(R,xyz).T    
    
    # Go back to spherical coordinates
    new_lat = np.arccos(rotated_points[:,2]).reshape(h,w)
    new_lon = np.arctan2(rotated_points[:,1],rotated_points[:,0]).reshape(h,w)
    neg = np.where(new_lon<0)
    new_lon[neg] += 2*np.pi
    
    # Remap image
    y_map = new_lat * h / np.pi
    x_map = new_lon * w / (2 * np.pi)
    new_image = cv2.remap(image, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_NEAREST, 0, cv2.BORDER_REPLICATE)   #cv2.INTER_CUBIC

    return new_image




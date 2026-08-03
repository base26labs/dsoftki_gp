"""MD22 dataset download.

If you use MD22, cite:

    Chmiela, S., Vassilev-Galindo, V., Unke, O. T., Kabylda, A., Sauceda, H. E.,
    Tkatchenko, A., & Mueller, K.-R. (2023). Accurate global machine learning
    force fields for molecules with hundreds of atoms. Science Advances, 9(2),
    eadf0873. https://doi.org/10.1126/sciadv.adf0873
"""

from os.path import exists
import requests
import tarfile
from zipfile import ZipFile 

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


STANDARDIZE = True
COULOMB = False

# def torch_dist_mat(x):
#     num_points = x.size(0)
#     tensor1 = x.unsqueeze(1).expand(-1, num_points, -1)
#     tensor2 = x.unsqueeze(0).expand(num_points, -1, -1)
#     distance_matrix = torch.linalg.vector_norm(tensor1 - tensor2, dim=2, ord=1)
#     return distance_matrix


def coulomb_desc(zs, x):
    A = len(zs)
    radii = torch.cdist(x.unsqueeze(0), x.unsqueeze(0))
    radii[:, torch.arange(A), torch.arange(A)] = 1
    radii = radii.squeeze(0).clip(0, 1e6)
    M = zs.unsqueeze(1) @ zs.unsqueeze(0)
    M = M / radii
    M[torch.arange(A), torch.arange(A)] = 0.5*zs**2.4
    upper_tri_indices = torch.triu_indices(M.size(0), M.size(1), offset=0)
    upper_triangular_flat = M[upper_tri_indices[0], upper_tri_indices[1]]
    return upper_triangular_flat


def to_unit_cube(x, lb, ub, g=None):
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub, g=None):
    xx = x * (ub - lb) + lb
    return xx


# =============================================================================
# Base Dataset
# =============================================================================

class MD22Dataset(Dataset):
    def __init__(self, npz_file="./md22/md22_DHA.npz", dtype=torch.float32, transform=None, standarize=False, flat=False, center=True, coulomb=False, get_forces=False, unit_cube=False):
        # Store arguments
        self.npz_file = npz_file
        self.dtype = dtype
        self.get_forces = get_forces
        self.transform = transform
        self.center = center
    
        # Unpack data
        self.raw_data = np.load(npz_file)
        self.coords = []
        self.energies = []
        self.forces = []
        for x, e, f in tqdm(zip(self.raw_data["R"], self.raw_data["E"], self.raw_data["F"])):
            if coulomb:
                self.coords += [coulomb_desc(torch.tensor(self.raw_data["z"].flatten()).to(dtype), torch.tensor(x).to(dtype))]
            else:
                self.coords += [torch.tensor(x.flatten())]
            if flat:
                self.energies += [e[0]]
            else:
                self.energies += [e]
            self.forces += [torch.tensor(f.flatten(), dtype=dtype)]
        self.coords = torch.stack(self.coords).to(dtype=dtype)
        self.energies = torch.tensor(self.energies).to(dtype=dtype)  # negate the energy instead of the force
        self.forces = torch.stack(self.forces)
        self.zs = torch.tensor(self.raw_data["z"].flatten()).to(dtype)
        self.dim = len(self.coords[0].reshape(-1))
        print(self.energies.shape, self.forces.shape)

        # Standardize data
        self.shift = 0
        self.scale = 1
        if standarize:
            # Standardize x units
            if unit_cube:
                mins = [x.min() for x in self.coords]
                maxs = [x.max() for x in self.coords]
                lb = np.array(mins).min()
                ub = np.array(maxs).max()
                print("lb ub", lb, ub)
                self.lb = lb
                self.ub = ub
                # self.x_scale = ub - lb
                # self.coords = self.coords / self.x_scale
                self.coords = torch.tensor(to_unit_cube(self.coords, lb, ub).to(dtype=dtype))
            elif False:
                scaler = StandardScaler()
                self.coords = torch.tensor(scaler.fit_transform(self.coords)).to(dtype=dtype)
            # elif False:
            #     self.coords = (self.coords - lb) / self.scale
            else:
                self.x_scale = 3
                self.coords = self.coords / self.x_scale

            # Standardize y units            
            # sigma = torch.std(self.energies, unbiased=True)
            # self.energies /= sigma
            mu = torch.mean(self.energies)  # Negated energy
            print("mean", mu.shape, self.energies.shape, mu)
            sigma = torch.std(torch.cat([(-self.energies + mu).unsqueeze(-1), self.forces], dim=-1))  # Adjust force (but not energy)
            print("energy shapes", self.energies.shape, "force shape", self.forces.shape)
            print("separate mean std", torch.std(self.energies), "separate force std", torch.std(self.forces))
            self.energies = (-self.energies + mu) / sigma
            print("mean", mu, "std", sigma, "scale")
            self.forces /= sigma
            
            self.shift = mu
            self.scale = sigma
        else:
            if self.center:
                self._center_energies()

    def _center_energies(self) -> None:
        self.mean = self.energies.mean()
        self.energies = self.energies - self.mean

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.coords[idx]
        if self.transform:
            features = self.transform(features)

        if self.get_forces:
            return features, {
                "energy": self.energies[idx],
                "neg_force": self.forces[idx],
            }
        else:
            label = self.energies[idx]
            return features, label
    

# =============================================================================
# AcAla3NHME
# =============================================================================

def get_AcAla3NHME():
    if not exists('./md22/md22_Ac-Ala3-NHMe.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_Ac-Ala3-NHMe.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_Ac-Ala3-NHMe.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_AcAla3NHME_Dataset(npz_file="./md22/md22_Ac-Ala3-NHMe.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_AcAla3NHME_Dataset(MD22Dataset):
    """
    N = 85109
    D = 42 x 3 = 126
    """    
    def __init__(self, npz_file="./md22/md22_Ac-Ala3-NHMe.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_AcAla3NHME_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)


# =============================================================================
# Docosahexaenoic acid
# =============================================================================

def get_DHA():
    if not exists('./md22/md22_DHA.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_DHA.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_DHA.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_DHA.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_DHA_Dataset(npz_file="./md22/md22_DHA.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DHA_Dataset(MD22Dataset):
    """
    N = 69753
    D = 56 x 3 = 168
    """    
    def __init__(self, npz_file="./md22/md22_DHA.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_DHA_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)
        

# =============================================================================
# Stachyose
# =============================================================================

def get_stachyose():
    if not exists('./md22/md22_stachyose.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_stachyose.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_stachyose.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_stachyose.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_Stachyose_Dataset(npz_file="./md22/md22_stachyose.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_Stachyose_Dataset(MD22Dataset):
    """
    N = 27272
    D = 87 x 3 = 261
    """    
    def __init__(self, npz_file="./md22/md22_stachyose.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_Stachyose_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)
    

# =============================================================================
# AT-AT
# =============================================================================

def get_dna_at_at():
    if not exists('./md22/md22_AT-AT.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_AT-AT.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_AT-AT.npz")
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DNA_AT_AT_Dataset(npz_file="./md22/md22_AT-AT.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DNA_AT_AT_Dataset(MD22Dataset):
    """
    N = 20001
    D = 60 x 3 = 180
    """    
    def __init__(self, npz_file="./md22/md22_AT-AT.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_DNA_AT_AT_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)


# =============================================================================
# AT-AT
# =============================================================================

def get_dna_at_at_cg_cg():
    if not exists('./md22/md22_AT-AT-CG-CG.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_AT-AT-CG-CG.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_AT-AT-CG-CG.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DNA_AT_AT_Dataset(npz_file="./md22/md22_AT-AT-CG-CG.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass

    M = 10
    A = len(dataset.zs)
    subset = np.stack([np.random.choice(A, 4, replace=False) for _ in range(M)])
    sections = np.concatenate([3*subset, 3*subset+1, 3*subset+2], axis=1)
    indices = torch.from_numpy(sections)
    features, labels = zip(*[(torch.tensor(features), torch.tensor(labels)) for features, labels in dataset])
    features = torch.stack(features).squeeze(-1)
    labels = torch.stack(labels).squeeze(-1)
    dataset2 = CoulombMD22Dataset(dataset.zs, features, labels, subset, indices)
    dataloader = DataLoader(dataset2, batch_size=1024)
    for x, y in tqdm(dataloader):
        print(x.shape, y.shape)


class MD22_DNA_AT_AT_CG_CG_Dataset(MD22Dataset):
    """
    N = 10153
    D = 118 x 3 = 354
    """    
    def __init__(self, npz_file="./md22/md22_AT-AT-CG-CG.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_DNA_AT_AT_CG_CG_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)


# =============================================================================
# Buckyball Catcher
# =============================================================================

def get_buckyball_catcher():
    if not exists('./md22/md22_buckyball-catcher.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_buckyball-catcher.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_buckyball-catcher.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_Buckyball_Catcher_Dataset(npz_file="./md22/md22_buckyball-catcher.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass

    M = 10
    A = len(dataset.zs)
    subset = np.stack([np.random.choice(A, 4, replace=False) for _ in range(M)])
    sections = np.concatenate([3*subset, 3*subset+1, 3*subset+2], axis=1)
    indices = torch.from_numpy(sections)
    features, labels = zip(*[(torch.tensor(features), torch.tensor(labels)) for features, labels in dataset])
    features = torch.stack(features).squeeze(-1)
    labels = torch.stack(labels).squeeze(-1)
    dataset2 = CoulombMD22Dataset(dataset.zs, features, labels, subset, indices)
    dataloader = DataLoader(dataset2, batch_size=1024)
    for x, y in tqdm(dataloader):
        print(x.shape, y.shape)


class MD22_Buckyball_Catcher_Dataset(MD22Dataset):
    """
    N = 6102
    D = 148 x 3 = 444
    """    
    def __init__(self, npz_file="./md22/md22_buckyball-catcher.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_Buckyball_Catcher_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=False, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)


# =============================================================================
# Double-Walled Nanotub
# =============================================================================

def get_double_walled_nanotube():
    if not exists('./md22/md22_double-walled_nanotube.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_double-walled_nanotube.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_double-walled_nanotube.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DoubleWalledNanotube_Dataset(npz_file="./md22/md22_double-walled_nanotube.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass

    M = 10
    A = len(dataset.zs)
    subset = np.stack([np.random.choice(A, 4, replace=False) for _ in range(M)])
    sections = np.concatenate([3*subset, 3*subset+1, 3*subset+2], axis=1)
    indices = torch.from_numpy(sections)
    features, labels = zip(*[(torch.tensor(features), torch.tensor(labels)) for features, labels in dataset])
    features = torch.stack(features).squeeze(-1)
    labels = torch.stack(labels).squeeze(-1)
    dataset2 = CoulombMD22Dataset(dataset.zs, features, labels, subset, indices)
    dataloader = DataLoader(dataset2, batch_size=1024)
    for x, y in tqdm(dataloader):
        print(x.shape, y.shape)


class MD22_DoubleWalledNanotube_Dataset(MD22Dataset):
    """
    N = 5032
    D = 370 x 3 = 1110
    """    
    def __init__(self, npz_file="./md22/md22_double-walled_nanotube.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, coulomb=COULOMB, get_forces=False, unit_cube=False):
        super(MD22_DoubleWalledNanotube_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, coulomb=coulomb, get_forces=get_forces, unit_cube=unit_cube)



class CoulombMD22Dataset(Dataset):
    def __init__(self, zs, features, labels, subset, indices) -> None:
        super().__init__()
        self.features = features
        self.labels = labels
        self.zs = zs
        self.subset = subset
        self.indices = indices.cpu()
        self.features2 = []
        for x in self.features:
            pi_x = torch.gather(x.unsqueeze(0).expand(self.indices.shape[0], -1), 1, self.indices)
            self.features2 += [self.coulomb_desc(self.zs[subset], pi_x.reshape(self.indices.shape[0], -1, 3))]
        self.features2 = torch.stack(self.features2)

    def coulomb_desc(self, zs, x):
        A = zs.shape[-1]
        radii = torch.cdist(x, x)
        diag = torch.eye(zs.shape[-1]).unsqueeze(0) * (2.0 * zs ** (-2.4)).unsqueeze(-1)
        return (1.0 / ((1.0 / zs.unsqueeze(-1)) * torch.sqrt(radii + 1e-10) + diag)).reshape(-1, A * A)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features2[idx]
        label = self.labels[idx]
        return features, label



if __name__ == "__main__":
    # Create md22 directory if it doesn't exist
    import os
    os.makedirs('./md22', exist_ok=True)

    get_AcAla3NHME()
    get_DHA()
    get_stachyose()
    get_dna_at_at()
    get_dna_at_at_cg_cg()
    get_buckyball_catcher()
    get_double_walled_nanotube()

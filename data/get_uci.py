import numpy as np
import os
import pandas as pd
import subprocess
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# Datasets
# =============================================================================

all_datasets = {
    "3droad": (434874, 3),
    "autompg": (392, 7),
    "bike": (17379, 17),
    "challenger": (23, 4),
    "concreteslump": (103, 7),
    "energy": (768, 8),
    "forest": (517, 12),
    "houseelectric": (2049280, 11),
    "keggdirected": (48827, 20),
    "kin40k": (40000, 8),
    "parkinsons": (5875, 20),
    "pol": (15000, 26),
    "pumadyn32nm": (8192, 32),
    "slice": (53500, 385),
    "solar": (1066, 10),
    "stock": (536, 11),
    "yacht": (308, 6),
    "airfoil": (1503, 5),
    "autos": (159, 25),
    "breastcancer": (194, 33),
    "buzz": (583250, 77),
    "concrete": (1030, 8),
    "elevators": (16599, 18),
    "fertility": (100, 9),
    "gas": (2565, 128),
    "housing": (506, 13),
    "keggundirected": (63608, 27),
    "machine": (209, 7),
    "pendulum": (630, 9),
    "protein": (45730, 9),
    "servo": (167, 4),
    "skillcraft": (3338, 19),
    "sml": (4137, 26),
    "song": (515345, 90),
    "tamielectric": (45781, 3),
    "wine": (1599, 11),
}


datasets = [
    "pol",
    "elevators",
    "bike",
    "kin40k",
    "protein",
    "keggdirected",
    "slice",
    "keggundirected",
    "3droad",
    "song",
    "buzz",
    "houseelectric",
]


# =============================================================================
# UCI Dataset
# =============================================================================

class UCIDataset(Dataset):
    def __init__(
        self,
        csv_file="./foobar.csv",
        dim=1,
        transform=None,
        standardize=True,
        minmax=False, 
        header=True,
        sep=None,
        get_forces=False,
        n_neighbors=5,
        use_gradient_cache=True
    ):
        """
        Args:
            csv_file (str): Path to the CSV file.
            dim (int): Number of feature columns. The label is assumed to be the next column.
            transform (callable, optional): Optional transform to be applied on a sample.
            standardize (bool, optional): Whether to standardize the features and label.
            header (bool, optional): Whether the CSV file has a header row.
            sep (str, optional): Delimiter for the CSV file.
            get_forces (bool, optional): Whether to compute and return gradients.
            n_neighbors (int, optional): Number of neighbors for gradient computation.
            use_gradient_cache (bool, optional): Whether to cache computed gradients.
        """
        self.get_forces = get_forces
        self.n_neighbors = n_neighbors
        self.use_gradient_cache = use_gradient_cache
        
        # Create cache file path
        cache_dir = os.path.dirname(csv_file)
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        self.gradient_cache_file = os.path.join(cache_dir, f"{dataset_name}_gradients_{n_neighbors}nn.npz")
        # Load data using pandas
        if sep is not None:
            self.raw_data = pd.read_csv(csv_file, sep=sep)
        elif header:
            self.raw_data = pd.read_csv(csv_file)
        else:
            self.raw_data = pd.read_csv(csv_file, header=None)
        
        print("SIZE", self.raw_data.shape)
        self.transform = transform
        self.dim = dim

        # Preprocess data
        self._preprocess(standardize, minmax)
        
        # Load or compute gradients if requested
        if self.get_forces:
            self._load_or_compute_gradients()

    def _preprocess(self, standardize: bool, minmax: bool) -> None:
        """
        Preprocesses the data by standardizing all columns (features and label)
        if required, then splits them into separate arrays.
        """
        if standardize:
            if minmax:
                scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            else:
                scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.raw_data)
            # Re-wrap into a DataFrame so that columns remain aligned
            self.data = pd.DataFrame(scaled_data, columns=self.raw_data.columns)
        else:
            self.data = self.raw_data

        # Split into features and label, then convert to float32
        self.features = self.data.iloc[:, :self.dim].values.astype(np.float32)
        self.labels = self.data.iloc[:, self.dim].values.astype(np.float32)
        
    def _load_or_compute_gradients(self):
        """Load cached gradients or compute them using k-nearest neighbors"""
        
        # Try to load cached gradients
        if self.use_gradient_cache and os.path.exists(self.gradient_cache_file):
            try:
                cached_data = np.load(self.gradient_cache_file)
                self.gradients = cached_data['gradients']
                print(f"✓ Loaded cached gradients from {self.gradient_cache_file}")
                print(f"  Gradient shape: {self.gradients.shape}")
                return
            except Exception as e:
                print(f"Failed to load cached gradients: {e}, recomputing...")
        
        # Compute gradients from scratch
        print(f"Computing gradients using {self.n_neighbors} nearest neighbors...")
        self.gradients = self._compute_gradients()
        
        # Save to cache
        if self.use_gradient_cache:
            try:
                os.makedirs(os.path.dirname(self.gradient_cache_file), exist_ok=True)
                np.savez_compressed(self.gradient_cache_file, 
                                   gradients=self.gradients)
                print(f"✓ Saved gradients to {self.gradient_cache_file}")
            except Exception as e:
                print(f"Failed to save gradients: {e}")
                
        print(f"✓ Gradient computation complete. Shape: {self.gradients.shape}")
        
    def _compute_gradients(self):
        """Compute gradients using k-nearest neighbor finite differences (optimized)"""
        gradients = np.zeros_like(self.features)
        
        # Use faster algorithm for high-D data and reduce neighbors for speed
        algorithm = 'auto' if self.dim < 20 else 'brute'
        
        # Fit k-nearest neighbors - get all at once (vectorized)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm=algorithm).fit(self.features)
        
        # Get all neighbors at once (much faster than one-by-one)
        print(f"Finding {self.n_neighbors} neighbors for all {len(self.features)} points...")
        all_distances, all_indices = nbrs.kneighbors(self.features)
        
        print("Computing gradients in vectorized batches...")
        for i in tqdm(range(len(self.features)), desc="Computing gradients"):
            # Skip self (first neighbor)
            neighbor_indices = all_indices[i][1:]
            neighbor_distances = all_distances[i][1:]
            
            # Vectorized computation
            valid_mask = neighbor_distances > 1e-8
            if np.any(valid_mask):
                valid_neighbors = neighbor_indices[valid_mask]
                valid_distances = neighbor_distances[valid_mask]
                
                # Vectorized direction computation
                directions = self.features[valid_neighbors] - self.features[i]
                direction_norms = np.linalg.norm(directions, axis=1)
                
                # Filter out zero-norm directions
                nonzero_mask = direction_norms > 1e-8
                if np.any(nonzero_mask):
                    directions = directions[nonzero_mask]
                    direction_norms = direction_norms[nonzero_mask]
                    valid_neighbor_indices = valid_neighbors[nonzero_mask]
                    
                    # Unit directions
                    unit_directions = directions / direction_norms[:, np.newaxis]
                    
                    # Target differences
                    target_diffs = self.labels[valid_neighbor_indices] - self.labels[i]
                    
                    # Gradient contributions (vectorized)
                    gradient_contribs = (target_diffs[:, np.newaxis] / direction_norms[:, np.newaxis]) * unit_directions
                    
                    # Sum and average
                    gradients[i] = np.mean(gradient_contribs, axis=0)
                
        return gradients.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, label) or (features, {"energy": label, "neg_force": gradient})
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        feature = self.features[idx]
        label = self.labels[idx]

        feature = torch.from_numpy(feature)
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            feature = self.transform(feature)


        if self.get_forces:
            gradient = torch.from_numpy(self.gradients[idx])
            return feature, {
                "energy": self.labels[idx],  # Keep as tensor like MD22
                "neg_force": gradient
            }
        else:
            return feature, label


class PoleteleDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/pol/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(PoleteleDataset, self).__init__(csv_file=csv_file, dim=26, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class ElevatorsDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/elevators/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(ElevatorsDataset, self).__init__(csv_file=csv_file, dim=18, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class BikeDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/bike/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(BikeDataset, self).__init__(csv_file=csv_file, dim=17, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class Kin40KDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/kin40k/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(Kin40KDataset, self).__init__(csv_file=csv_file, dim=8, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class ProteinDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/protein/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(ProteinDataset, self).__init__(csv_file=csv_file, dim=9, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class KeggDirectedDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/keggdirected/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(KeggDirectedDataset, self).__init__(csv_file=csv_file, dim=20, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class CTSlicesDataset(UCIDataset):
    """CT Slice Localization - D=385 (HIGH-D)"""
    def __init__(self, csv_file="./uci_datasets/uci_datasets/slice/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(CTSlicesDataset, self).__init__(csv_file=csv_file, dim=385, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class KeggUndirectedDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/keggundirected/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(KeggUndirectedDataset, self).__init__(csv_file=csv_file, dim=27, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class RoadDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/3droad/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(RoadDataset, self).__init__(csv_file=csv_file, dim=3, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class SongDataset(UCIDataset):
    """Million Song Dataset - D=90 (HIGH-D)"""
    def __init__(self, csv_file="./uci_datasets/uci_datasets/song/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(SongDataset, self).__init__(csv_file=csv_file, dim=90, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class BuzzDataset(UCIDataset):
    """Buzz Social Media - D=77 (HIGH-D)"""
    def __init__(self, csv_file="./uci_datasets/uci_datasets/buzz/data.csv", transform=None, standardize=True, minmax=False, get_forces=False, n_neighbors=5):
        super(BuzzDataset, self).__init__(csv_file=csv_file, dim=77, transform=transform, standardize=standardize, minmax=minmax, header=False, get_forces=get_forces, n_neighbors=n_neighbors)


class HouseElectricDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/houseelectric/data.csv", transform=None, standardize=True, minmax=False):
        super(HouseElectricDataset, self).__init__(csv_file=csv_file, dim=11, transform=transform, standardize=standardize, minmax=minmax, header=False)


# =============================================================================
# Main
# =============================================================================

def test_gradient_datasets():
    """Test high-dimensional UCI datasets with gradients"""
    print("Testing High-Dimensional UCI Datasets with Gradients")
    print("=" * 55)
    
    # High-D datasets to test
    high_d_datasets = [
        ("Buzz", BuzzDataset, 77),
        ("Song", SongDataset, 90),
        ("CTSlices", CTSlicesDataset, 385), 
    ]
    
    for name, dataset_class, expected_dim in high_d_datasets:
        print(f"\nTesting {name} dataset (D={expected_dim})...")
        try:
            # Test with gradients
            dataset = dataset_class(get_forces=True, n_neighbors=3)
            
            # Test sample
            x, y = dataset[0]
            print(f"✓ {name}: {len(dataset)} samples")
            print(f"  Input shape: {x.shape}")
            print(f"  Energy: {y['energy']:.4f}")
            print(f"  Gradient shape: {y['neg_force'].shape}")
            print(f"  Dimensions match: {x.shape[0] == y['neg_force'].shape[0]}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test_gradients":
        test_gradient_datasets()
    else:
        # Original UCI dataset setup
        if not os.path.exists("./uci_datasets"):
            print("trying to clone")
            subprocess.run(["git", "clone", "https://github.com/treforevans/uci_datasets.git"], shell=False)
        from pathlib import Path
        base_dir = Path(__file__).parent / "uci_datasets" / "uci_datasets"

        for dataset in datasets:
            gz_file_path = os.path.join(base_dir, dataset, "data.csv.gz")
            if os.path.exists(gz_file_path):
                print(f"Unzipping {dataset}")
                subprocess.run(["gzip", "-d", gz_file_path])
            else:
                print(f"No gzip file found for {dataset} at {gz_file_path}")
                
        print(f"\nTo test gradient computation, run:")
        print(f"  python data/get_uci.py test_gradients")
    
    torch_datasets = [
        PoleteleDataset,
        ElevatorsDataset,
        BikeDataset,
        Kin40KDataset,
        ProteinDataset,
        KeggDirectedDataset,
        CTSlicesDataset,
        KeggUndirectedDataset,
        RoadDataset,
        SongDataset,
        BuzzDataset,
        # HouseElectricDataset,
    ]

    for torch_dataset in torch_datasets:
        print(torch_dataset)
        # Enable gradients for datasets that support it
        if hasattr(torch_dataset, '__init__'):
            try:
                dataset = torch_dataset(get_forces=True, n_neighbors=5)
                print(f"✓ Dataset with gradients: {len(dataset)} samples")
            except:
                dataset = torch_dataset()
                print(f"✓ Dataset without gradients: {len(dataset)} samples")
        else:
            dataset = torch_dataset()
            
        dataloader = DataLoader(dataset, batch_size=1024)
        for x, y in tqdm(dataloader):
            if isinstance(y, dict):
                # print(f"  Input: {x.shape}, Energy: {y['energy'].shape}, Gradients: {y['neg_force'].shape}")
                assert x.shape == y['neg_force'].shape
            else:
                # print(f"  Input: {x.shape}, Target: {y.shape}")
                pass
            # break  # Just test first batch
        
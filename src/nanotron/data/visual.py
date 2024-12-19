from pyarrow.parquet import ParquetDataset

class RawParquetFolderDataset(Dataset):
    def __init__(self, path: str, transform: Optional[Callable] = None):
        self.path = path
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(path, "*.parquet")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        df = pd.read_parquet(file)
        if self.transform:
            df = self.transform(df)
        return df
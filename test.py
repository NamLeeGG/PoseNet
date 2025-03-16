from src.data.datamodule import DataModule

def test_dataloader():
    dm = DataModule()
    dm.setup()
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    imgs, heatmaps = batch
    assert imgs.shape == (32, 3, 256, 128), "Bad image batch shape"
    assert heatmaps.shape == (32, 24, 64, 32), "Bad heatmap batch shape"
    assert heatmaps.min() >= 0 and heatmaps.max() <= 1, "Invalid heatmap values"
    
    print("Dataloader validated!")

if __name__ == "__main__":
    test_dataloader()
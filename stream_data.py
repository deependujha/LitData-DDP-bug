from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset("data")
dataloader = StreamingDataLoader(
    dataset,
    batch_size=4,
    drop_last=True,
)

for batch in dataloader:
    print(batch["output"])
    print("-"*80)

import torch
import litdata as ld


def random_data(index):
    fake_input = torch.randn(1)
    fake_output = torch.randn(1)

    # You can use any key:value pairs. Note that their types must not change between samples, and Python lists must
    # always contain the same number of elements with the same types.
    data = {"input": fake_input, "output": fake_output, "idx": index}

    return data


if __name__ == "__main__":
    # The optimize function writes data in an optimized format.
    ld.optimize(
        fn=random_data,  # the function applied to each input
        inputs=list(
            range(100)
        ),  # the inputs to the function (here it's a list of numbers)
        output_dir="data",  # optimized data is stored here
        num_workers=4,  # The number of workers on the same machine
        chunk_bytes="64MB",  # size of each chunk
        mode="overwrite",  # overwrite existing data
    )

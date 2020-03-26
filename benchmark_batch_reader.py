import time
import numpy as np
from petastorm import TransformSpec, make_batch_reader
from petastorm.pytorch import DataLoader
from petastorm.unischema import UnischemaField


def _print(*args):
    print("OUTPUT", *args)


# def _transform_row(row):
#     from PIL import Image
#     from petastorm.codecs import CompressedImageCodec
#     from torchvision import transforms
#     import time
#     import pyarrow as pa
#
#     transform = transforms.Compose(
#         [
#             # transforms.RandomResizedCrop(224),
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     )
#
#     # import pdb; pdb.set_trace()
#
#     codec = CompressedImageCodec("png")
#     t0 = time.time()
#
#     column = row.column("image")
#     # new_column = pa.array(
#     #     [
#     #         transform(Image.fromarray(codec.decode("image", column[i].as_buffer())))
#     #         .flatten()
#     #         .numpy()
#     #         for i in range(column.shape[0])
#     #     ]
#     # )
#     # row_new = row.set_column(0, pa.Column.from_array("image", new_column))
#
#     a = [(codec.decode("image", column[i].as_buffer())) for i in range(column.shape[0])]
#
#     dt = time.time() - t0
#     print("processing: {:.4g}".format(dt / row.shape[0]))
#     return row

def _transform_row(row):
  from PIL import Image
  from petastorm.codecs import CompressedImageCodec
  from torchvision import transforms

  transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ])
  codec = CompressedImageCodec('png')
  row['image'] = row['image'].apply(
    lambda row: transform(Image.fromarray(codec.decode("image", row))).numpy())

  return row


transform = TransformSpec(_transform_row)


transform = TransformSpec(_transform_row, edit_fields=[('image', np.uint8, (3, 224, 224), False),])


def main():
    for worker_count in [1]: #[10, 4, 1]:
        _print("testing make_reader with number of workers", worker_count)
        with make_batch_reader(
            "file:///home/yevgeni/temp/fardin/train",
            num_epochs=None,
            reader_pool_type="dummy",
            workers_count=worker_count,
            cur_shard=1,
            shard_count=4,
            hdfs_driver="libhdfs",
            schema_fields=["image"],
            transform_spec=transform,
            shuffle_row_groups=False,
        ) as reader:

            batch_size = 128
            iterations = 250

            #   loader = DataLoader(reader,
            #                       batch_size=batch_size,
            #                       shuffling_queue_capacity=0)
            loader = reader

            it = iter(loader)

            for _ in range(4):
                next(it)

            #
            # # Warmup
            # # for _ in range(3):
            # #   a = next(it)
            # _print("Done warming up")
            #
            # count = 0
            # tstart = time.time()
            # for i in range(iterations):
            #     if i % 20 == 0:
            #         _print("step", i)
            #     a = next(it)
            #     # import pdb; pdb.set_trace()
            #     count += a.image.shape[0]
            #
            #     _print(
            #         "make_batch_reader, worker_count: {}, Samples per "
            #         "second for batch {}: {:.4g}".format(
            #             worker_count, batch_size, (count) / (time.time() - tstart)
            #         )
            #     )
            #

if __name__ == "__main__":
    main()

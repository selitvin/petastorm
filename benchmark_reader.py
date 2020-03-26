import time

from petastorm import TransformSpec, make_reader
from petastorm.pytorch import DataLoader


def _print(*args):
    print("OUTPUT", *args)


def _transform_row(row):
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import cv2

    if not hasattr(np, 'transform_def'):
        np.transform_def = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])

    transform_def = np.transform_def

    row['image'] = np.asarray(transform_def(Image.fromarray(row['image'])))
    # row['image'] = np.asarray(transform_def(Image.fromarray(row['image'])))

    #   row['image'] = np.asarray(Image.fromarray(row['image']))
    #   import pdb; pdb.set_trace()
    #   row['image'] = cv2.imdecode(np.frombuffer(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return row


transform = TransformSpec(_transform_row)


def main():
    for worker_count in [1]:  # [10, 4, 1]:
        _print("testing make_reader with number of workers", worker_count)
        with make_reader(
                # 'file:///mnt/share/fardin/imagenet/imagenet_medium_parquet/train',
                "file:///home/yevgeni/temp/fardin/train",
                num_epochs=1,
                reader_pool_type='dummy',
                workers_count=worker_count,
                cur_shard=1,
                shard_count=4,
                hdfs_driver='libhdfs',
                schema_fields=['image'],
                transform_spec=transform,
                pyarrow_serialize=True,
            shuffle_row_groups=False,
        ) as reader:

            print("Done")
            batch_size = 128
            iterations = 250

            # loader = DataLoader(reader,
            #                     batch_size=batch_size,
            #                     shuffling_queue_capacity=0)
            loader = reader
            print("Done creating data loader")
            it = iter(loader)

            for i, _ in enumerate(reader):
                pass
            print("Done epoch")
            # Warmup
            # for _ in range(50):
            #     a = next(it)
            #     # import pdb; pdb.set_trace()
            # _print("Done warming up")
            #
            # count = 0
            # tstart = time.time()
            # for i in range(iterations * batch_size):
            #     sample = next(it)
            #     count += 1  # sample['image'].shape[0]
            #
            #     if i % 20 == 0:
            #         _print("make_reader, num_iterations:{} worker_count: {}, batch size: {}, Samples per "
            #                "second: {:.4g}".format(iterations, worker_count, batch_size,
            #                                        (count) / (time.time() - tstart)))


if __name__ == "__main__":
    main()

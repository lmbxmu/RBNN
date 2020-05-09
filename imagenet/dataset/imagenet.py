"""Dali dataloader for imagenet"""
import math
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class HybridPipe(Pipeline):
    def __init__(self, train, batch_size, num_threads, device_id, data_dir, crop, local_rank=0, world_size=1):
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12+device_id)
        self.input = ops.FileReader(
            file_root=data_dir,
            random_shuffle=train,
            shard_id=local_rank,
            num_shards=world_size,
        )
        if train:
            self.decode = ops.ImageDecoderRandomCrop(
                output_type=types.RGB,
                device="mixed",
                random_aspect_ratio=[0.75, 1.25],
                random_area=[0.08, 1.25],
                num_attempts=100,
            )
            self.resize = ops.Resize(
                device="gpu",
                interp_type=types.INTERP_TRIANGULAR,
                resize_x=crop,
                resize_y=crop,
            )
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            crop_size = math.ceil((crop * 1.14 + 8) // 16 * 16)
            self.resize = ops.Resize(
                device="gpu", interp_type=types.INTERP_TRIANGULAR, resize_shorter=crop_size,
            )
        # color augs
        self.contrast = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")

        self.jitter = ops.Jitter(device="gpu")
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            image_type=types.RGB,
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
        )
        self.coin = ops.CoinFlip(probability=0.5)

        self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.85, 1.15])
        self.rng3 = ops.Uniform(range=[-15, 15])
        self.train = train

    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        images = self.resize(images)
        if self.train:
            images = self.contrast(images, contrast=self.rng2(), brightness=self.rng2())
            images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())
            images = self.jitter(images, mask=self.coin())
            images = self.normalize(
                images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1()
            )
        else:
            images = self.normalize(images)
        return images, labels

def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop,
                           world_size=1,
                           local_rank=0):
    if type=='train':
        pipe = HybridPipe(train=True,batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                        data_dir=image_dir + '/ILSVRC2012_img_train',
                                        crop=crop, world_size=world_size, local_rank=local_rank)
    else: 
        pipe = HybridPipe(train=False,batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                        data_dir=image_dir + '/val',
                                        crop=crop, world_size=world_size, local_rank=local_rank)
    pipe.build()
    dataloader = DALIClassificationIterator(
        pipe,
        size=pipe.epoch_size("Reader") / world_size,
        auto_reset=True
    )
    return dataloader
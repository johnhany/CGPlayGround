import math
import time

import numpy as np
import taichi as ti
import torch

from PIL import Image


######## PyTorch ########

device = 'cuda'

tile_width = 32
tile_height = 48
pw = 4096
ph = 2048
sx = 10
N = 8

shift_x = torch.tensor([tile_width, 0], device=device)
shift_y = torch.tensor([sx, tile_height], device=device)

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph

image_pixels = torch.zeros((image_height, image_width, 3),
                           device=device,
                           dtype=float)
nrows = torch.arange(image_height, device=device)
ncols = torch.arange(image_width, device=device)
cy, cx = torch.meshgrid(ncols, nrows)
coords = torch.stack((cx.T, cy.T), axis=2)


def save_to_image(image_pixels, name):
    r = torch.sin(math.pi * image_pixels)
    g = torch.cos(math.pi * image_pixels / 2)
    b = torch.zeros_like(image_pixels)
    image_pixels = torch.stack([r, g, b], axis=2)
    image_pixels = (255 * image_pixels.cpu().numpy()).astype(np.uint8)
    Image.fromarray(image_pixels).save(f"output/{name}.png")


def generate_tile_image(tile_height, tile_width, device='cuda'):
    tile = torch.arange(0,
                        tile_width * tile_height,
                        dtype=torch.float,
                        device=device).reshape(tile_width, tile_height)
    tile = torch.rot90(tile, -1, (0, 1)).contiguous()
    tile = tile / (tile_height * tile_width)
    save_to_image(tile, 'tile_padding_origin')
    return tile

tile = generate_tile_image(tile_height, tile_width)
y = torch.tensor(tile.stride(), device=device, dtype=torch.float)


def torch_pad(arr, tile, y):
    # image_pixel_to_coord
    arr[:, :, 0] = image_height - 1 + ph - arr[:, :, 0]
    arr[:, :, 1] -= pw
    arr1 = torch.flip(arr, (2, ))

    # map_coord
    v = torch.floor(arr1[:, :, 1] / tile_height).to(torch.int)
    u = torch.floor((arr1[:, :, 0] - v * shift_y[0]) / tile_width).to(torch.int)
    uu = torch.stack((u, u), axis=2)
    vv = torch.stack((v, v), axis=2)
    arr2 = arr1 - uu * shift_x - vv * shift_y

    # coord_to_tile_pixel
    arr2[:, :, 1] = tile_height - 1 - arr2[:, :, 1]
    table = torch.flip(arr2, (2, ))
    table = table.view(-1, 2).to(torch.float)
    inds = table.mv(y)
    gathered = torch.index_select(tile.view(-1), 0, inds.to(torch.long))

    return gathered


class Timer:
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *exc_info):
        print(f'Elapsed {time.time() - self.start} seconds')


# Warmup: PyTorch code dramatically slows down when GPU RAM hits its limit 
# so it actually need a bit tweak to find the best runs.
for _ in range(3):
    gathered = torch_pad(coords, tile, y)
    gathered.zero_()

# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
for _ in range(N):
    with Timer():
        gathered = torch_pad(coords, tile, y)
        torch.cuda.synchronize(device=device)
# print(prof.key_averages().table(sort_by="self_cuda_time_total"))

save_to_image(gathered.reshape(image_height, image_width), 'tile_padding_torch')


######## Taichi ########

ti.init(arch=ti.gpu, kernel_profiler=True)
torch_device = 'cuda'

shift_x = (tile_width, 0)
shift_y = (sx, tile_height)

tile = generate_tile_image(tile_height, tile_width, torch_device)

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph

image_pixels = torch.zeros((image_height, image_width),
                           device=torch_device,
                           dtype=torch.float)

@ti.kernel
def ti_pad(image_pixels: ti.types.ndarray(), tile: ti.types.ndarray()): # type: ignore
    for row, col in ti.ndrange(image_height, image_width):
        # image_pixel_to_coord: maps a pixel in the large image to its coordinates (x, y) relative to the origin.
        # (which is the lower left corner of the tile image)
        x1, y1 = ti.math.ivec2(col - pw, image_height - 1 - row + ph)
        # map_coord: let (x, y) = (x0, y0) + shift_x * u + shift_y * v.
        # This function finds P = (x0, y0) which is the corresponding point in the tile.
        v: ti.i32 = ti.floor(y1 / tile_height) # type: ignore
        u: ti.i32 = ti.floor((x1 - v * shift_y[0]) / tile_width) # type: ignore
        x2, y2 = ti.math.ivec2(x1 - u * shift_x[0] - v * shift_y[0],
                 y1 - u * shift_x[1] - v * shift_y[1])
        # coord_to_tile_pixel: The origin is located at the lower left corner of the tile image.
        # Assuming a point P with coordinates (x, y) lies in this tile, this function
        # maps P's coordinates to its actual pixel location in the tile image.
        x, y = ti.math.ivec2(tile_height - 1 - y2, x2)
        image_pixels[row, col] = tile[x, y]


# Run once to compile pad kernel
ti_pad(image_pixels, tile)
image_pixels.fill_(0)

ti.profiler.clear_kernel_profiler_info()
for _ in range(N):
    with Timer():
        ti_pad(image_pixels, tile)
        ti.sync()
ti.profiler.print_kernel_profiler_info()

save_to_image(image_pixels, 'tile_padding_taichi')

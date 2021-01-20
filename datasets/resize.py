from PIL import Image
import os, sys
import argparse
from pathlib import Path


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dir', default='road2icy/trainB/')
arg_parser.add_argument('--output_dir', default='road2icy-64/trainB/')
arg_parser.add_argument('--size', default=256, type=int)
opt = arg_parser.parse_args()


def resize(opt):
    path = opt.dir
    dirs = os.listdir( path )
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    print(str(output_dir))
    for item in dirs:
        print(path+item)
        if os.path.isfile(path+item):
            print(i)
            i = i+1
            im = Image.open(path+item).convert("RGB")
            f, e = os.path.splitext(path+item)
            imResize = im.resize((opt.size, opt.size), Image.ANTIALIAS)
            imResize.save(str(output_dir) + "/" + str(i) + '.jpg', 'JPEG', quality=90)

resize(opt)
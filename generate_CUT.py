import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from train import save_image
import util.util as util


"""
Single-Scale Unaligned Dataset
Use with CUTGAN Implementation
"""
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batch_size = 1
    dataset = create_dataset(opt)
    model = create_model(opt)

    for i, data in enumerate(dataset):
        if i==0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        for label, image in visuals.items():
            if label == "fake_B":
                image_numpy = util.tensor2im(image)
                img_path = os.path.join("datasets/generated", '%.3d_image.png' % i)
                save_image(image_numpy, img_path)
    


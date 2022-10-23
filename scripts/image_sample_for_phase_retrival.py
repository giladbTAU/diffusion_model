"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import scipy.io as io
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision as tv
import cv2

from guided_diffusion.gaussian_diffusion import _extract_into_tensor

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


    # np_src_image = np.asarray(Image.open(args.y_image))
    convert_tensor = transforms.ToTensor()
    mat = io.loadmat(args.y_image)  #2048x2048
    mat = mat['amp_matrix']
    mat = np.flip(mat,0).astype(np.float32)
  
    print("image_sample_for_phase_retrival: mat.shape", mat.shape)
    print("image_sample_for_phase_retrival: mat.min", mat.min())
    print("image_sample_for_phase_retrival: mat.max", mat.max())    
    y_mat_data = th.tensor(mat).unsqueeze(0).to(device)    
    # y_mat_data = convert_tensor(Image.open(args.y_image)).cuda()      
    # y_mat_data = th.from_numpy(transforms.ToTensor()(np.array(Image.open(args.y_image))))
    y_mat_data_flipped = th.flip(y_mat_data, [0])
    print("image_sample_for_phase_retrival: y_mat_data_flipped.shape", y_mat_data_flipped.shape)

    # y_mat_data_flipped = y_mat_data_flipped.permute(0,3,1,2) # just for jpg image
    y_mat_data_flipped = y_mat_data_flipped.repeat(args.batch_size,1,1,1)
    print("image_sample_for_phase_retrival: y_mat_data_flipped.shape", y_mat_data_flipped.shape)
    print("image_sample_for_phase_retrival: y_mat_data_flipped.min", y_mat_data_flipped.min())
    print("image_sample_for_phase_retrival: y_mat_data_flipped.max", y_mat_data_flipped.max()) 
    print("YYYYYYYYYYYYYYYYYYYYYYY")

# Look here for likelihood
    def cond_fn(cond_fn_inputs, t, y=None):  # Amer and Gilad: calculate the gradients
        """
        Amer and Gilad: from page 8 in the article, we are calculating the s*gradient log p(y|x)
        s = args.classifier_scale
        gradient = th.autograd.grad(selected.sum(), x_in)[0]
        """
        x, eps, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod = cond_fn_inputs
        assert y is not None
        with th.enable_grad():             
  
            x_in = x.detach().requires_grad_(True)
            # print("image_sample_for_phase_retrival: x_in.shape", x_in.shape)
            # print("image_sample_for_phase_retrival: x_in.min", x_in.min())
            # print("image_sample_for_phase_retrival: x_in.max", x_in.max()) 
            # print("image_sample_for_phase_retrival: x_in.type", x_in.type)
            x_0 = _extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_in.shape) * x_in
            - _extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_in.shape) * eps
            """
            Amer and Gilad: FIXME: need to change the below lines so this function will return the new "g" as Shady explained
            """
            print("TTTTTTTTTTTTTTTTTTT")
            new_x_in_flipped = th.zeros(x_in.shape[0],x_in.shape[1],y_mat_data.shape[1],y_mat_data.shape[2]).to(device) #16,3,2048,2048
            #th.pad FIXME
            x_in_flipped = th.flip(x_0, [2])
            new_x_in_flipped[:,:,945:1201, 945:1201] = x_in_flipped
            new_x_in = th.flip(new_x_in_flipped, [2])
            a_x = th.fft.ifftshift(th.fft.fft2(th.fft.fftshift(new_x_in), norm="backward")) #no normaliztion
            # a_x /= float(a_x.shape[2] * a_x.shape[3])
            a_x = th.abs(a_x)**2
            print("image_sample_for_phase_retrival: a_x.shape", a_x.shape)
            print("image_sample_for_phase_retrival: a_x.min", a_x.min())
            print("image_sample_for_phase_retrival: a_x.max", a_x.max())            
            loss = th.pow(th.abs((y_mat_data_flipped-a_x)/(2048**2)), 2)  # loss = MSE
            loss = loss.mean()
            #################### Shady:
            # try to do sum and not mean.
            # scale factor
            # add minus to g
            print("loss:", loss)
            print("ZZZZZZZZZZZZZZ")         
            # loss = l1 + l2 + l3 + l4 -> l1/dx1, l2/dx2, ...   : each x_in is affecting its corresponding loss
            # l1 - log probability of the first image
            # we are doing a gradient with respect to x_in
            g = -th.autograd.grad(loss, new_x_in)[0] * args.classifier_scale
            print("image_sample_for_phase_retrival: g.shape", g.shape)
            print("image_sample_for_phase_retrival: g.min", g.min())
            print("image_sample_for_phase_retrival: g.max", g.max())             
            # rescale back to 256x256
            g_flipped = th.flip(g, [2])
            new_g_flipped = g_flipped[:, :, 945:1201, 945:1201]
            new_g = th.flip(new_g_flipped, [2])
            return new_g
            # return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_series = []
    all_img_series = []
    cond_fn_en = (cond_fn if args.gradient_guidance else None)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, _ , _, _ = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            # cond_fn=None,
            cond_fn=(cond_fn if args.gradient_guidance else None),
            device=dist_util.dev(),
        )
        print("image_sample_for_phase_retrival: sample.shape:", sample.shape)
        print("image_sample_for_phase_retrival: sample.min:", sample.min())
        print("image_sample_for_phase_retrival: sample.max:", sample.max())
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]


        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        print(len(gathered_samples))  # tensor type
        print(gathered_samples[0].shape)        
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(len(all_images))  # numpy ndarray type
        print(all_images[0].shape)
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        model_name = os.path.basename(args.model_path).replace('.pt','')
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{model_name}_take2_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_head_channels=64,
        learn_sigma=True,
        use_scale_shift_norm=True,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        class_cond=False,
        gradient_guidance = False,
        classifier_scale=10.0,
        classifier_path="guided_diffusion/article_example_models/256x256_classifier.pt",
        model_path="guided_diffusion/article_example_models/256x256_diffusion_uncond.pt",
        y_image=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

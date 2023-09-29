from PIL import Image
import numpy as np
import cv2
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from cldm.ddim_hacked import PLMSSampler as DDIMSampler
import mindspore as ms
from mindspore import ops
import os
import argparse
import time


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

apply_canny = CannyDetector()

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    print(f'Loaded model config from [{config_path}]')
    return model


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def load_state_dict(model, path='torch2ms/ms_weight'):
    print(f"Loading model from {path}")

    unet_weight = ms.load_checkpoint(os.path.join(path, 'unet.ckpt'))
    ms.load_param_into_net(model.model, unet_weight)

    vae_weight = ms.load_checkpoint(os.path.join(path, 'vae.ckpt'))
    ms.load_param_into_net(model.first_stage_model, vae_weight)

    text_encoder_weight = ms.load_checkpoint(os.path.join(path, 'text_encoder.ckpt'))
    ms.load_param_into_net(model.cond_stage_model, text_encoder_weight)

    controlnet_weight = ms.load_checkpoint(os.path.join(path, 'controlnet.ckpt'))
    ms.load_param_into_net(model.control_model, controlnet_weight)
    
    return model


def load_model(config_path, pretrained_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    model = load_state_dict(model, path=pretrained_path)
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler


def process(input_path, low_threshold=100, high_threshold=200, image_resolution=512):
    input_image = Image.open(input_path).convert('RGB')
    input_image = np.asarray(input_image)
    img = resize_image(HWC3(input_image), image_resolution)
    img = input_image
    H, W, C = img.shape

    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)

    return img, detected_map


def inference(control, config_path, pretrained_path,
              prompt, n_prompt, num_samples, 
              ddim_steps, guess_mode, strength, scale, eta, 
              width=512, height=512):
    
    control_map = control.copy()

    # load model
    model, ddim_sampler = load_model(config_path=config_path, pretrained_path=pretrained_path)

    # process control map
    control = np.transpose(control.copy(), (2, 0, 1))
    control = ms.Tensor(control.copy()) / 255.0
    control = ops.stack([control for _ in range(num_samples)], axis=0)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, height // 8, width // 8)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = ops.transpose(x_samples, (0, 2, 3, 1)) * 127.5 + 127.5
    x_samples = x_samples.asnumpy().copy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return [255 - control_map] + results


def save(results, output_path):
    control = results[0]
    samples = results[1:]

    dt_string = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(f'{output_path}/{dt_string}', exist_ok=True)
    
    Image.fromarray(control).save(f'{output_path}/{dt_string}/control.png')
    for i in range(len(samples)):
        Image.fromarray(samples[i]).save(f'{output_path}/{dt_string}/sample_{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', default='torch2ms/ms_weight', type=str)
    parser.add_argument('--config_path', default='configs/cldm_v15.yaml', type=str)
    parser.add_argument('--input_path', default=None, type=str)
    parser.add_argument(
        '--prompt', 
        default='a girl,best quality,extremely detailed', 
        type=str
    )
    parser.add_argument(
        '--negative_prompt', 
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 
        type=str
    )
    parser.add_argument('--output_path', default='output/controlnet', type=str)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--image_resolution', default=512, type=int)
    parser.add_argument('--ddim_steps', default=20, type=int)
    parser.add_argument('--guess_mode', default=False, type=bool)
    parser.add_argument('--strength', default=1.0, type=float)
    parser.add_argument('--scale', default=9.0, type=float)
    parser.add_argument('--eta', default=0.0, type=float)

    args = parser.parse_args()

    
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target="GPU",
        device_id=device_id,
        max_device_memory="30GB"
    )

    _, control_map = process(args.input_path)
    results = inference(control_map, args.config_path, args.pretrained_path,
                        args.prompt, args.negative_prompt, args.num_samples, 
                        args.ddim_steps, args.guess_mode, args.strength, args.scale, args.eta)
    
    save(results, args.output_path)

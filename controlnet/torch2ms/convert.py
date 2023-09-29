import numpy as np
import mindspore as ms
import pickle
import torch
import argparse


def convert_torch_to_numpy(path, name='cond_stage_model', save=False):
    # name: cond_stage_model, diffusion_model, first_stage_model, control_model
    # path = '/mnt/petrelfs/majie/project/ControlNet/models/control_sd15_canny.pth'
    torch_weight = torch.load(path)

    numpy_weight = {}
    for k, v in torch_weight.items():
        if name in k:
            numpy_weight[k] = v.numpy()
    
    if save:
        with open(f'./numpy_weight/{name}.pkl', 'wb') as f:
            pickle.dump(numpy_weight, f)
    return numpy_weight


def save_ms_ckpt(ckpt, output_dir, name):
    save_data = []
    for k, v in ckpt.items():
        save_data.append({
            'name': k,
            'data': v
        })

    ms.save_checkpoint(save_data, f'{output_dir}/{name}.ckpt')


def convert_text_encoder(numpy_weight = {}):
    # 'embeddings.position_ids': None,
    transformer_layers = {}
    transformer_layers[-1] = {}
    for i in range(12):
        transformer_layers[i] = {}

    for k, v in numpy_weight.items():
        if 'encoder.layers' in k:
            layer_num = int(k.split('.')[5])
            transformer_layers[layer_num][k] = v
        else:
            transformer_layers[-1][k] = v

    for i in range(12):
        keys = list(transformer_layers[i].keys())

        # self_attn
        q_weight = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.q_proj.weight'.format(i)]
        q_bias = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.q_proj.bias'.format(i)]
        k_weight = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.k_proj.weight'.format(i)]
        k_bias = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.k_proj.bias'.format(i)]
        v_weight = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.v_proj.weight'.format(i)]
        v_bias = transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.v_proj.bias'.format(i)]

        qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
        qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)

        transformer_layers[i]['cond_stage_model.transformer.transformer_layer.resblocks.{}.attn.attn.in_proj.weight'.format(i)] = qkv_weight
        transformer_layers[i]['cond_stage_model.transformer.transformer_layer.resblocks.{}.attn.attn.in_proj.bias'.format(i)] = qkv_bias

        # layer norm
        transformer_layers[i]['cond_stage_model.transformer.transformer_layer.resblocks.{}.attn.attn.out_proj.weight'.format(i)] = \
            transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.out_proj.weight'.format(i)]
        transformer_layers[i]['cond_stage_model.transformer.transformer_layer.resblocks.{}.attn.attn.out_proj.bias'.format(i)] = \
            transformer_layers[i]['cond_stage_model.transformer.text_model.encoder.layers.{}.self_attn.out_proj.bias'.format(i)]

        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.ln_1.gamma'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.weight']
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.ln_1.beta'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.bias']
        
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.ln_2.gamma'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.weight']
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.ln_2.beta'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.bias']
        
        # mlp
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.c_fc.weight'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.weight']
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.c_fc.bias'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.bias']
        
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.c_proj.weight'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.weight']
        transformer_layers[i][f'cond_stage_model.transformer.transformer_layer.resblocks.{i}.c_proj.bias'] = \
            transformer_layers[i][f'cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.bias']

        # remove origin keys
        for key in keys:
            del transformer_layers[i][key]

    transformer_layers[-1]['cond_stage_model.transformer.embedding_table'] = \
        transformer_layers[-1]['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    transformer_layers[-1]['cond_stage_model.transformer.positional_embedding'] = \
        transformer_layers[-1]['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']
    transformer_layers[-1]['cond_stage_model.transformer.ln_final.gamma'] = \
        transformer_layers[-1]['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    transformer_layers[-1]['cond_stage_model.transformer.ln_final.beta'] = \
        transformer_layers[-1]['cond_stage_model.transformer.text_model.final_layer_norm.bias']
    
    # remove origin key
    del transformer_layers[-1]['cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']
    del transformer_layers[-1]['cond_stage_model.transformer.text_model.embeddings.position_embedding.weight']
    del transformer_layers[-1]['cond_stage_model.transformer.text_model.final_layer_norm.weight']
    del transformer_layers[-1]['cond_stage_model.transformer.text_model.final_layer_norm.bias']

    final_ckpt = {}
    for i in range(12):
        for k, v in transformer_layers[i].items():
            final_ckpt[k] = v
    
    for k, v in transformer_layers[-1].items():
        final_ckpt[k] = v
    
    # convert numpy to mindspore
    for k, v in final_ckpt.items():
        final_ckpt[k] = ms.Tensor(v)

    return final_ckpt


def convert_vae(numpy_weight = {}):
    change = {
        'norm1.weight': 'norm1.gamma',
        'norm1.bias': 'norm1.beta',

        'norm2.weight': 'norm2.gamma',
        'norm2.bias': 'norm2.beta',

        'norm.weight': 'norm.gamma',
        'norm.bias': 'norm.beta',

        'norm_out.weight': 'norm_out.gamma',
        'norm_out.bias': 'norm_out.beta',
    }
    
    final_ckpt = {}
    for k, v in numpy_weight.items():
        for old_key, new_key in change.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
                final_ckpt[k] = ms.Tensor(v)
                continue
        final_ckpt[k] = ms.Tensor(v)
    
    return final_ckpt
    

def convert_unet(numpy_weight={}):
    ms_weight = {}

    change = {
        'input_blocks.0.0.weight': 'input_blocks.0.0.conv.weight',
        'input_blocks.0.0.bias': 'input_blocks.0.0.conv.bias',

        'in_layers.0.weight': 'in_layers_norm.gamma', 
        'in_layers.0.bias': 'in_layers_norm.beta',
        'in_layers.2.weight': 'in_layers_conv.conv.weight',
        'in_layers.2.bias': 'in_layers_conv.conv.bias',

        'out_layers.0.weight': 'out_layers_norm.gamma',
        'out_layers.0.bias': 'out_layers_norm.beta',
        'out_layers.3.weight': 'out_layers_conv.conv.weight',
        'out_layers.3.bias': 'out_layers_conv.conv.bias',

        'op.weight': 'op.conv.weight',
        'op.bias': 'op.conv.bias',

        'norm.weight': 'norm.gamma',
        'norm.bias': 'norm.beta',

        'norm1.weight': 'norm1.gamma',
        'norm1.bias': 'norm1.beta',

        'norm2.weight': 'norm2.gamma',
        'norm2.bias': 'norm2.beta',

        'norm3.weight': 'norm3.gamma',
        'norm3.bias': 'norm3.beta',

        'skip_connection': 'skip_connection.conv',

        'output_blocks.2.1.conv.weight': 'output_blocks.2.1.conv.conv.weight',
        'output_blocks.2.1.conv.bias': 'output_blocks.2.1.conv.conv.bias',

        'output_blocks.5.2.conv.bias': 'output_blocks.5.2.conv.conv.bias',
        'output_blocks.5.2.conv.weight': 'output_blocks.5.2.conv.conv.weight',

        'output_blocks.8.2.conv.bias': 'output_blocks.8.2.conv.conv.bias',
        'output_blocks.8.2.conv.weight': 'output_blocks.8.2.conv.conv.weight',

        '.out.0.weight': '.out.0.gamma',
        '.out.0.bias': '.out.0.beta',
        '.out.2.weight': '.out.2.conv.weight',
        '.out.2.bias': '.out.2.conv.bias',

    }

    for k, v in numpy_weight.items():
        for old_key, new_key in change.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
                ms_weight[k] = ms.Tensor(v)
                continue
        ms_weight[k] = ms.Tensor(v)
    
    return ms_weight


def convert_controlnet(numpy_weight = {}):
    change = {
        'input_blocks.0.0.weight': 'input_blocks.0.0.conv.weight',
        'input_blocks.0.0.bias': 'input_blocks.0.0.conv.bias',

        'in_layers.0.weight': 'in_layers_norm.gamma',
        'in_layers.0.bias': 'in_layers_norm.beta',
        'in_layers.2.weight': 'in_layers_conv.conv.weight',
        'in_layers.2.bias': 'in_layers_conv.conv.bias',

        'out_layers.0.weight': 'out_layers_norm.gamma',
        'out_layers.0.bias': 'out_layers_norm.beta',
        'out_layers.3.weight': 'out_layers_conv.conv.weight',
        'out_layers.3.bias': 'out_layers_conv.conv.bias',

        'norm.weight': 'norm.gamma',
        'norm.bias': 'norm.beta',
        'norm1.weight': 'norm1.gamma',
        'norm1.bias': 'norm1.beta',
        'norm2.weight': 'norm2.gamma',
        'norm2.bias': 'norm2.beta',
        'norm3.weight': 'norm3.gamma',
        'norm3.bias': 'norm3.beta',

        'op.weight': 'op.conv.weight',
        'op.bias': 'op.conv.bias',

        'skip_connection': 'skip_connection.conv',
    }

    final_ckpt = {}
    for k, v in numpy_weight.items():
        if 'zero_convs' in k or 'input_hint_block' in k or 'middle_block_out' in k:
            k = k.replace('.weight', '.conv.weight')
            k = k.replace('.bias', '.conv.bias')
        else:
            for old_key, new_key in change.items():
                if old_key in k:
                    k = k.replace(old_key, new_key)
                    break
        final_ckpt[k] = ms.Tensor(v)

    return final_ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--only_controlnet', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default=None)
    
    args = parser.parse_args()

    if args.only_controlnet:
        controlnet = convert_torch_to_numpy(path=args.input_path, name='control_model')
        save_ms_ckpt(convert_controlnet(controlnet), output_dir=args.output_path, name='controlnet')

    else:
        vae = convert_torch_to_numpy(path=args.input_path, name='first_stage_model')
        text_encoder = convert_torch_to_numpy(path=args.input_path, name='cond_stage_model')
        unet = convert_torch_to_numpy(path=args.input_path, name='diffusion_model')
        controlnet = convert_torch_to_numpy(path=args.input_path, name='control_model')

        save_ms_ckpt(convert_vae(vae), output_dir=args.output_path, name='vae')
        save_ms_ckpt(convert_text_encoder(text_encoder), output_dir=args.output_path, name='text_encoder')
        save_ms_ckpt(convert_unet(unet), output_dir=args.output_path, name='unet')
        save_ms_ckpt(convert_controlnet(controlnet), output_dir=args.output_path, name='controlnet')

    print('Done!')


import mindspore as ms
import mindspore.nn as nn

from ldm.modules.diffusionmodules.openaimodel import UNetModel, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import exists, instantiate_from_config


class ControlledUnetModel(UNetModel):
    def construct(self, x, timesteps, context, 
                  control_1, control_2, control_3, control_4, control_5, control_6):
        
        control = []
        if control_1 is not None:
            control = ms.ops.split(control_1, axis=0, output_num=control_1.shape[0]) \
                    + ms.ops.split(control_2, axis=0, output_num=control_2.shape[0]) \
                    + ms.ops.split(control_3, axis=0, output_num=control_3.shape[0]) \
                    + ms.ops.split(control_4, axis=0, output_num=control_4.shape[0]) \
                    + ms.ops.split(control_5, axis=0, output_num=control_5.shape[0]) \
                    + ms.ops.split(control_6, axis=0, output_num=control_6.shape[0])
            control = list(control)
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.astype(self.dtype)

        for module in self.input_blocks:
            for cell in module:
                h = cell(h, emb, context)
            hs.append(h)
        
        for module in self.middle_block:
            h = module(h, emb, context)

        control_idx = -1
        if len(control) != 0:
            h += control[control_idx]
            control_idx -= 1
        
        only_mid_control = False
        hs_idx = -1
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or len(control) == 0:
                h = ms.ops.concat([h, hs[hs_idx].astype(h.dtype)], axis=1)
            else:
                h = ms.ops.concat([h, hs[hs_idx].astype(h.dtype) + control[control_idx].astype(h.dtype)], axis=1)
                control_idx -= 1
                hs_idx -= 1
            for cell in module:
                h = cell(h, emb, context)

        return self.out(h)
    

class ControlNet(nn.Cell):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=1.0,  # may contain bug; is different with pytorch
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")
        

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.CellList([
                nn.CellList([conv_nd(dims, in_channels, model_channels, 3, padding=1, 
                            has_bias=True, pad_mode='pad').to_float(self.dtype)])
            ])
        
        self.zero_convs = nn.CellList([self.make_zero_conv(model_channels)])

        self.input_hint_block = nn.SequentialCell(
            conv_nd(dims, hint_channels, 16, 3, padding=1, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2, has_bias=True, pad_mode='pad'),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1, has_bias=True, pad_mode='pad'))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.CellList([
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype
                    )
                ])
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            use_checkpoint=use_checkpoint, dtype=self.dtype, dropout=self.dropout
                        )
                    )

                self.input_blocks.append(layers)
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.CellList(
                        [ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype
                        )])
                        if resblock_updown
                        else nn.CellList([Downsample(ch, conv_resample, 
                                            dims=dims, out_channels=out_ch, dtype=self.dtype)])
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch
        
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels


        self.middle_block =  nn.CellList([
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype
                    ),
                    AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=dim_head,
                        use_new_attention_order=use_new_attention_order,
                    ) if not use_spatial_transformer else SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    use_checkpoint=use_checkpoint, dtype=self.dtype, dropout=self.dropout,
                                ),
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype
                    ),
                ])
        
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return nn.SequentialCell([
            zero_module(
                conv_nd(self.dims, channels, channels, 1, padding=0, has_bias=True, pad_mode='pad').to_float(self.dtype)
            )
        ])

    def construct(self, x, hint, timesteps, context):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint)
        
        outs = []

        h = x.astype(self.dtype)

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                for cell in module:
                    h = cell(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                for cell in module:
                    h = cell(h, emb, context)

            outs.append(zero_conv(h))
        
        for module in self.middle_block:
            h = module(h, emb, context)

        outs.append(self.middle_block_out(h))

        return outs
    

class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model

        cond_txt = ms.ops.concat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, 
                                  control_1=None, control_2=None,
                                  control_3=None, control_4=None,
                                  control_5=None, control_6=None,
                                  )
        else:
            control = self.control_model(x=x_noisy, hint=ms.ops.concat(cond['c_concat'], 1), timesteps=t, 
                                         context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            
            control_1 = ms.ops.concat(control[:3], 0)
            control_2 = control[3]
            control_3 = ms.ops.concat(control[4:6], 0)
            control_4 = control[6]
            control_5 = ms.ops.concat(control[7:9], 0)
            control_6 = ms.ops.concat(control[9: ])


            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, 
                                  control_1=control_1, control_2=control_2,
                                  control_3=control_3, control_4=control_4,
                                  control_5=control_5, control_6=control_6,
                                  )
        return eps

    def get_input(self, x, c, control):
        x, c = super().get_input(x, c)
        
        control = ms.numpy.transpose(control, (0, 3, 1, 2))

        return x, c, control

    def construct(self, x, c, control):
        t = ms.ops.UniformInt()((x.shape[0],), ms.Tensor(0, dtype=ms.dtype.int32), ms.Tensor(self.num_timesteps, dtype=ms.dtype.int32))
        x, c, control = self.get_input(x, c, control)
        c = self.get_learned_conditioning_fortrain(c)
        return self.p_losses(x, c, t, control)

    def p_losses(self, x_start, cond, t, control, noise=None):
        noise = ms.numpy.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        tmp = {'c_concat': [control], 'c_crossattn': [cond]}
        model_output = self.apply_model(x_noisy, t, tmp)
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ms.ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss += (self.original_elbo_weight * loss_vlb)
        
        return loss
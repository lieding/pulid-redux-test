import types
import comfy
import math

from PulidFluxHook import set_model_dit_patch_replace, PatchKeys, pulid_forward_orig, add_model_patch_option

from comfy import model_management
import torch
from torch import Tensor
from torch import nn

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        # if mask is not None:
            # not sure
            # weight.shape (bs, n_heads, seq_len, seq_len)
            # mask.shape (bs, seq_len, _) -> (bs, 1, 1, seq_len)
            # mask = mask[:,:, :1].view(b, 1, 1, -1)
            # weight = weight.masked_fill(mask == 0, float('-inf'))
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)

class PulidFluxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.double_interval = 2
        self.single_interval = 4

        # Init attention
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA() for _ in range(num_ca)
        ])

    def from_pretrained(self, path: str):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

wrappers_name = "PULID_wrappers"

def apply_pulid_flux(model, pulid_flux, cond: Tensor, weight: float, start_at: float, end_at: float, attn_mask=None):
    model = model.clone()

    # device = comfy.model_management.get_torch_device()
    # Why should I care what args say, when the unet model has a different dtype?!
    # Am I missing something?!
    #dtype = comfy.model_management.unet_dtype()
    dtype = model.model.diffusion_model.dtype
    # Because of 8bit models we must check what cast type does the unet uses
    # ZLUDA (Intel, AMD) & GPUs with compute capability < 8.0 don't support bfloat16 etc.
    # Issue: https://github.com/balazik/ComfyUI-PuLID-Flux/issues/6
    if model.model.manual_cast_dtype is not None:
        dtype = model.model.manual_cast_dtype

    pulid_flux.model.to(dtype=dtype)
    model_management.load_models_gpu([pulid_flux], force_full_load=True)


    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "pulid_ca": pulid_flux.model.pulid_ca,
        "weight": weight,
        "embedding": cond,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "mask": attn_mask,
    }

    ca_idx = 0
    for i in range(19):
        if i % pulid_flux.model.double_interval == 0:
            patch_kwargs["ca_idx"] = ca_idx
            set_model_dit_patch_replace(model, patch_kwargs, ("double_block", i))
            ca_idx += 1
    for i in range(38):
        if i % pulid_flux.model.single_interval == 0:
            patch_kwargs["ca_idx"] = ca_idx
            set_model_dit_patch_replace(model, patch_kwargs, ("single_block", i))
            ca_idx += 1

    if len(model.get_additional_models_with_key("pulid_flux_model_patcher")) == 0:
        model.set_additional_models("pulid_flux_model_patcher", [pulid_flux])

    if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name)) == 0:
        # Just add it once when connecting in series
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, wrappers_name, pulid_outer_sample_wrappers_with_override)
    if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name)) == 0:
        # Just add it once when connecting in series
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrappers_name, pulid_apply_model_wrappers)

    return (model,)

def set_hook(diffusion_model, target_forward_orig):
    # comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid = comfy.ldm.flux.model.Flux.forward_orig
    # comfy.ldm.flux.model.Flux.forward_orig = pulid_forward_orig
    diffusion_model.old_forward_orig_for_pulid = diffusion_model.forward_orig
    diffusion_model.forward_orig = types.MethodType(target_forward_orig, diffusion_model)

def clean_hook(diffusion_model):
    # if hasattr(comfy.ldm.flux.model.Flux, 'old_forward_orig_for_pulid'):
    #     comfy.ldm.flux.model.Flux.forward_orig = comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    #     del comfy.ldm.flux.model.Flux.old_forward_orig_for_pulid
    if hasattr(diffusion_model, 'old_forward_orig_for_pulid'):
        diffusion_model.forward_orig = diffusion_model.old_forward_orig_for_pulid
        del diffusion_model.old_forward_orig_for_pulid


def pulid_apply_model_wrappers(wrapper_executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
    base_model = wrapper_executor.class_obj
    PULID_model_patch = transformer_options.get(PatchKeys.pulid_patch_key_attrs, {})
    PULID_model_patch['timesteps'] = base_model.model_sampling.timestep(t).float()
    try:
        transformer_options[PatchKeys.running_net_model] = base_model.diffusion_model
        out = wrapper_executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    finally:
        if PatchKeys.running_net_model in transformer_options:
            del transformer_options[PatchKeys.running_net_model]
        del PULID_model_patch['timesteps'], base_model

    return out

def pulid_outer_sample_wrappers_with_override(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = wrapper_executor.class_obj
    PULID_model_patch = add_model_patch_option(cfg_guider, PatchKeys.pulid_patch_key_attrs)
    PULID_model_patch['latent_image_shape'] = latent_image.shape

    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    set_hook(diffusion_model, pulid_forward_orig)
    try :
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        del PULID_model_patch['latent_image_shape']
        clean_hook(diffusion_model)
        del diffusion_model, cfg_guider

    return out



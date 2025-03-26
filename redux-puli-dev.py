import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from extra_config import load_extra_path_config as _load_extra_path_config
from nodes import NODE_CLASS_MAPPINGS, CheckpointLoaderSimple, LoadImage, CLIPTextEncode, EmptyLatentImage, VAEDecode, SaveImage, VAELoader, UNETLoader, KSampler, DualCLIPLoader
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance
import node_helpers
from apply_pulid import PulidFluxModel, apply_pulid_flux

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    err = False
    try:
        from main import load_extra_path_config
    except ImportError:
        err = True
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        if err:
            _load_extra_path_config(extra_model_paths)
        else:
            load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def zero_cond(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    return (c, )

add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

def load_pulid_model(pulid_file: str):
    # Also initialize the model, takes longer to load but then it doesn't have to be done every time you change parameters in the apply node
    offload_device = model_management.unet_offload_device()
    load_device = model_management.get_torch_device()

    model = PulidFluxModel()
    model.from_pretrained(path=pulid_file)

    model_patcher = ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    del model

    return (model_patcher,)


def main():
    # import_custom_nodes()
    loader = UNETLoader()
    model = loader.load_unet(unet_name="flux1-dev-fp8.safetensors", weight_dtype="default")
    vae_model = VAELoader().load_vae("ae.sft")
    model_loaders = [model, vae_model]
    model_management.load_models_gpu([
        loader[0].patcher if hasattr(loader[0], 'patcher') else loader[0] for loader in model_loaders
    ])
    pulid_model = load_pulid_model("pulid_flux_v0.9.1.safetensors")
    with torch.inference_mode():
        pulid_output = torch.load("pulid_outout_2025-03-26 21:53.pt")
        model = apply_pulid_flux(
            model=get_value_at_index(model, 0),
            pulid_flux=get_value_at_index(pulid_model, 0),
            cond=pulid_output["embedding"],
            weight=pulid_output["weight"],
            start_at=pulid_output["sigma_start"],
            end_at=pulid_output["sigma_end"],
            attn_mask=pulid_output["mask"]
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_37 = emptylatentimage.generate(
            width=512, height=896, batch_size=1
        )

        redux_output = torch.load("redux_cond_2025-03-26 18:35.pt")

        ksampler = KSampler().sample(
            model=get_value_at_index(model, 0),
            seed=random.randint(1, 2**64),
            steps=20,
            cfg=1,
            sampler_name="euler_ancestral",
            scheduler="normal",
            positive=redux_output,
            negative=get_value_at_index(zero_cond(redux_output), 0),
            latent_image=get_value_at_index(emptylatentimage_37, 0),
            denoise=1
        )

        vaedecode = VAEDecode()
        vaedecode_38 = vaedecode.decode(
            samples=get_value_at_index(ksampler, 0),
            vae=get_value_at_index(vae_model, 0),
        )

        saveimage = SaveImage()

        saveimage.save_images(
            filename_prefix="result", images=get_value_at_index(vaedecode_38, 0)
        )

if __name__ == "__main__":
    main()

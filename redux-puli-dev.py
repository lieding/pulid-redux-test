import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from comfy import model_management
from extra_config import load_extra_path_config as _load_extra_path_config
from nodes import NODE_CLASS_MAPPINGS, CheckpointLoaderSimple, LoadImage, CLIPTextEncode, EmptyLatentImage, VAEDecode, SaveImage, VAELoader, UNETLoader, KSampler
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_flux import FluxGuidance
import node_helpers

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



def main():
    loader = CheckpointLoaderSimple()
    model = loader.load_checkpoint(ckpt_name="flux-dev-fp8.safetensors")
    vae_model = VAELoader().load_vae("ae.sft")
    
    model_loaders = [model, vae_model, ]

    model_management.load_models_gpu([
        loader[0].patcher if hasattr(loader[0], 'patcher') else loader[0] for loader in model_loaders
    ])
    with torch.inference_mode():

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_37 = emptylatentimage.generate(
            width=512, height=896, batch_size=1
        )

        positive = torch.load("/home/featurize/work/pulid-redux-comfyui-deploy/positive.pt")
        negative = torch.load("/home/featurize/work/pulid-redux-comfyui-deploy/negative.pt")

        # dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        # dualcliploader_34 = dualcliploader.load_clip(
        #     clip_name1="t5xxl_fp16.safetensors", clip_name2="clip_l.safetensors", type="flux",
        # )
        # positive = CLIPTextEncode().encode(
        #     clip=get_value_at_index(dualcliploader_34, 0),
        #     text="This black-and-white photograph, likely taken with a high-resolution DSLR camera using a medium aperture (f/5.6), captures actor Hugh Jackman in a close-up portrait. Jackman, with his neatly combed, short hair, and a slight smile, wears a formal suit and tie. The lighting is dramatic, with a spotlight creating a halo effect around his face, casting shadows that highlight his facial features. The background is dark, emphasizing the subject. "
        # )
        # negative = CLIPTextEncode().encode(
        #     clip=get_value_at_index(dualcliploader_34, 0),
        #     text=""
        # )

        # torch.save(positive, "/home/featurize/work/pulid-redux-comfyui-deploy/positive.pt")
        # torch.save(negative, "/home/featurize/work/pulid-redux-comfyui-deploy/negative.pt")


        ksampler = KSampler().sample(
            model=get_value_at_index(model, 0),
            seed=random.randint(1, 2**64),
            steps=28,
            cfg=3.5,
            sampler_name="dpmpp_sde",
            scheduler="karras",
            positive=get_value_at_index(positive, 0),
            negative=get_value_at_index(negative, 0),
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


def _main():
    
    import_custom_nodes()
    with torch.inference_mode():
        loader = UNETLoader() # CheckpointLoaderSimple()
        #model = loader.load_checkpoint(ckpt_name="flux1-dev-fp8.safetensors")
        model = loader.load_unet(unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn")
        vaeload = VAELoader().load_vae("ae.sft")

        
        redux_output = torch.load("female_1_redux_prompt.pt")
        if False:
            dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            dualcliploader_34 = dualcliploader.load_clip(
                clip_name1="clip_l.safetensors", clip_name2="t5xxl_fp16.safetensors", type="flux",
            )
            encode = CLIPTextEncode().encode(
                clip=get_value_at_index(dualcliploader_34, 0),
                text="This black-and-white photograph, likely taken with a high-resolution DSLR camera using a medium aperture (f/5.6), captures actor Hugh Jackman in a close-up portrait. Jackman, with his neatly combed, short hair, and a slight smile, wears a formal suit and tie. The lighting is dramatic, with a spotlight creating a halo effect around his face, casting shadows that highlight his facial features. The background is dark, emphasizing the subject. "
            )
        
        ksamplerselect_35 = KSamplerSelect().get_sampler(sampler_name="euler")

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_37 = emptylatentimage.generate(
            width=512, height=896, batch_size=1
        )

       
        randomnoise_13 = (Noise_RandomNoise(random.randint(1, 2**64)),)

        fluxguidance_16 = FluxGuidance().append(
            guidance=3.5, conditioning=redux_output
        )

        basicguider_17 = BasicGuider().get_guider(
            model=get_value_at_index(model, 0),
            conditioning=get_value_at_index(fluxguidance_16, 0),
        )

        basicscheduler_36 = BasicScheduler().get_sigmas(
            scheduler="simple",
            steps=28,
            denoise=1,
            model=get_value_at_index(model, 0),
        )

        samplercustomadvanced_10 = SamplerCustomAdvanced().sample(
            noise=get_value_at_index(randomnoise_13, 0),
            guider=get_value_at_index(basicguider_17, 0),
            sampler=get_value_at_index(ksamplerselect_35, 0),
            sigmas=get_value_at_index(basicscheduler_36, 0),
            latent_image=get_value_at_index(emptylatentimage_37, 0),
        )
        vaedecode = VAEDecode()
        vaedecode_38 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_10, 0),
            vae=get_value_at_index(vaeload, 0),
        )

        saveimage = SaveImage()

        saveimage.save_images(
            filename_prefix="result", images=get_value_at_index(vaedecode_38, 0)
        )


if __name__ == "__main__":
    main()

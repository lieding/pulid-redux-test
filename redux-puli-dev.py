import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from nodes import NODE_CLASS_MAPPINGS
from comfy import model_management
from huggingface_hub import hf_hub_download
from extra_config import load_extra_path_config as _load_extra_path_config

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
    hf_hub_download(repo_id="Comfy-Org/flux1-dev", filename="flux1-dev-fp8.safetensors", cache_dir="models/checkpoints/")
    
    import_custom_nodes()
    with torch.inference_mode():
        CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]
        LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]
        PulidFluxModelLoader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]
        PulidFluxEvaClipLoader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]
        PulidFluxInsightFaceLoader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]
        ApplyPulidFlux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]

        checkpointLoaderSimple = CheckpointLoaderSimple()
        ckpt = checkpointLoaderSimple.load_checkpoint(
            ckpt_name="flux1-dev.safetensors"
        )
        loadimage = LoadImage()
        reference_loadimage = loadimage.load_image(image="PHOTO-2025-03-04-16-32-01.jpg")

        pulidfluxmodelloader = PulidFluxModelLoader()
        pulidfluxmodelloader_19 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )

        pulidfluxevacliploader = PulidFluxEvaClipLoader()
        pulidfluxevacliploader_21 = pulidfluxevacliploader.load_eva_clip()

        pulidfluxinsightfaceloader =PulidFluxInsightFaceLoader()
        pulidfluxinsightfaceloader_22 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        applypulidflux = ApplyPulidFlux()
        applypulidflux_20 = applypulidflux.apply_pulid_flux(
            weight=0.9500000000000001,
            start_at=0,
            end_at=1,
            fusion="mean",
            fusion_weight_max=1,
            fusion_weight_min=0,
            train_step=1000,
            use_gray=True,
            model=get_value_at_index(ckpt, 0),
            pulid_flux=get_value_at_index(pulidfluxmodelloader_19, 0),
            eva_clip=get_value_at_index(pulidfluxevacliploader_21, 0),
            face_analysis=get_value_at_index(pulidfluxinsightfaceloader_22, 0),
            image=get_value_at_index(reference_loadimage, 0),
            unique_id=1774265400529356555,
        )
        print(get_value_at_index(applypulidflux_20))


def __main():
    import_custom_nodes()
    with torch.inference_mode():
        RandomNoise = NODE_CLASS_MAPPINGS["RandomNoise"]
        UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]
        PulidFluxModelLoader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]
        PulidFluxEvaClipLoader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]
        PulidFluxInsightFaceLoader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]
        ApplyPulidFlux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]
        LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]
        StyleModelLoader = NODE_CLASS_MAPPINGS["StyleModelLoader"]
        CLIPVisionLoader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]
        

        randomnoise = RandomNoise()
        randomnoise_13 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        unetloader = UNETLoader()
        unetloader_18 = unetloader.load_unet(
            unet_name="基础算法_F.1", weight_dtype="fp8_e4m3fn"
        )

        loadimage = LoadImage()
        reference_loadimage = loadimage.load_image(image="PHOTO-2025-03-04-16-32-01.jpg")

        style_loadimage = loadimage.load_image(
            image="DUJARDIN-JEAN-AC-PP-7994-PS-03-A_LOGO-822x1024.jpg"
        )

        pulidfluxmodelloader = PulidFluxModelLoader()
        pulidfluxmodelloader_19 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )

        pulidfluxevacliploader = PulidFluxEvaClipLoader()
        pulidfluxevacliploader_21 = pulidfluxevacliploader.load_eva_clip()

        pulidfluxinsightfaceloader =PulidFluxInsightFaceLoader()
        pulidfluxinsightfaceloader_22 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        applypulidflux = ApplyPulidFlux()
        applypulidflux_20 = applypulidflux.apply_pulid_flux(
            weight=0.9500000000000001,
            start_at=0,
            end_at=1,
            fusion="mean",
            fusion_weight_max=1,
            fusion_weight_min=0,
            train_step=1000,
            use_gray=True,
            model=get_value_at_index(unetloader_18, 0),
            pulid_flux=get_value_at_index(pulidfluxmodelloader_19, 0),
            eva_clip=get_value_at_index(pulidfluxevacliploader_21, 0),
            face_analysis=get_value_at_index(pulidfluxinsightfaceloader_22, 0),
            image=get_value_at_index(reference_loadimage, 0),
            unique_id=1774265400529356555,
        )

        stylemodelloader = StyleModelLoader()
        stylemodelloader_28 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev"
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_29 = clipvisionloader.load_clip(
            clip_name="siglip-so400m-patch14-384"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_34 = dualcliploader.load_clip(
            clip_name1="clip_l", clip_name2="t5xxl_fp16", type="flux", device="default"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_32 = cliptextencode.encode(
            text="Ultra realistic. Masterpiece. This black-and-white photograph, likely taken with a high-resolution DSLR camera using a medium aperture (f/5.6), captures man in a close-up portrait. He is neatly combed, short hair, and a slight smile, wears a formal suit and tie. The lighting is dramatic, with a spotlight creating a halo effect around his face, casting shadows that highlight his facial features. The background is dark, emphasizing the subject. ",
            clip=get_value_at_index(dualcliploader_34, 0),
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_35 = ksamplerselect.get_sampler(sampler_name="euler")

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_37 = emptylatentimage.generate(
            width=768, height=1344, batch_size=1
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="ae.sft")


        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_50 = loraloadermodelonly.load_lora_model_only(
            lora_name="XLabs F.1 Realism LoRA_V1",
            strength_model=0.8,
            model=get_value_at_index(applypulidflux_20, 0),
        )

        imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        reduxadvanced = NODE_CLASS_MAPPINGS["ReduxAdvanced"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            imagescale_26 = imagescale.upscale(
                upscale_method="nearest-exact",
                width=768,
                height=1344,
                crop="center",
                image=get_value_at_index(style_loadimage, 0),
            )

            reduxadvanced_24 = reduxadvanced.apply_stylemodel(
                downsampling_factor=1,
                downsampling_function="nearest",
                mode="keep aspect ratio",
                weight=0.5,
                autocrop_margin=0.1,
                conditioning=get_value_at_index(cliptextencode_32, 0),
                style_model=get_value_at_index(stylemodelloader_28, 0),
                clip_vision=get_value_at_index(clipvisionloader_29, 0),
                image=get_value_at_index(imagescale_26, 0),
            )

            fluxguidance_16 = fluxguidance.append(
                guidance=3.5, conditioning=get_value_at_index(reduxadvanced_24, 0)
            )

            basicguider_17 = basicguider.get_guider(
                model=get_value_at_index(loraloadermodelonly_50, 0),
                conditioning=get_value_at_index(fluxguidance_16, 0),
            )

            basicscheduler_36 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=28,
                denoise=1,
                model=get_value_at_index(loraloadermodelonly_50, 0),
            )

            samplercustomadvanced_10 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_13, 0),
                guider=get_value_at_index(basicguider_17, 0),
                sampler=get_value_at_index(ksamplerselect_35, 0),
                sigmas=get_value_at_index(basicscheduler_36, 0),
                latent_image=get_value_at_index(emptylatentimage_37, 0),
            )

            vaedecode_38 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_10, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            saveimage_40 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_38, 0)
            )


if __name__ == "__main__":
    main()

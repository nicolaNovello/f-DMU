import os
import random
import shutil
from pathlib import Path
import numpy as np
import openai
import regex as re
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import transformers
import warnings
from diffusers import DPMSolverMultistepScheduler
from collections import defaultdict

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_anchor_ids = [example["instance_anchor_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    input_anchor_ids = torch.cat(input_anchor_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "input_anchor_ids": input_anchor_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1),
    }
    return batch

small_288 = transforms.Compose(
    [
        transforms.Resize(288),
        transforms.ToTensor(),
        normalize,
    ]
)



import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        concept_type,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
        anchor_type="superclass",
        class_prompt = "painting"
    ):
        self.anchor_type = anchor_type
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.LANCZOS
        self.aug = aug
        self.concept_type = concept_type
        
        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        self.class_prompt=class_prompt
        for concept in concepts_list:
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            inst_img_path = [
                (x, y, concept["caption_target"])
                for (x, y) in zip(inst_images_path, inst_prompt)
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [
                        concept["class_prompt"] for _ in range(len(class_images_path))
                    ]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [
                    (x, y) for (x, y) in zip(class_images_path, class_prompt)
                ]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.dict_prompts ={"van gogh_near" :"salvador dali",
                            "salvador dali_near" :"van gogh",
                            "r2d2_near" :"Wall-E",
                            "wall-e_near" :"r2d2",
                            "grumpy cat_near" :"snoopy",
                            "snoopy_near" :"grumpy cat"}
        

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(
            0, outer - inner + 1
        )
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // 8 + 1 : (top + scale) // 8 - 1,
                left // 8 + 1 : (left + scale) // 8 - 1,
            ] = 1.0
        return instance_image, mask

    def __getprompt__(self, instance_prompt, instance_target):
        if self.concept_type == "style":
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_prompt}, in the style of {instance_target}"
                if r == 0
                else f"in {instance_target}'s style, {instance_prompt}"
                if r == 1
                else f"in {instance_target}'s style, {instance_prompt}"
            )
        elif self.concept_type in ["nudity", "inappropriate_content"]:
            r = np.random.choice([0, 1, 2])
            instance_prompt = (
                f"{instance_target}, {instance_prompt}"
                if r == 0
                else f"in {instance_target} style, {instance_prompt}"
                if r == 1
                else f"{instance_prompt}, {instance_target}"
            )

        elif self.concept_type == "object":
            
            instance_prompt = instance_prompt.replace(self.class_prompt, instance_target)

        elif self.concept_type == "memorization":
            instance_prompt = instance_target.split("+")[1]
        return instance_prompt

    
    def __getitem__(self, index):
        example = {}
        
        instance_image, instance_prompt, instance_target = self.instance_images_path[
            index % self.num_instance_images
        ]

        instance_image = Image.open(instance_image)
  
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        
        if ";" in instance_target:
            instance_target = instance_target.split(";")
            instance_target = instance_target[index % len(instance_target)]
        
        if self.anchor_type == "superclass":
            instance_anchor_prompt = instance_prompt
            instance_prompt = self.__getprompt__(instance_prompt, instance_target)

        elif self.anchor_type == "empty":
            instance_anchor_prompt = ""#instance_prompt
            instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        elif self.anchor_type == "near":

            instance_anchor_prompt =self.__getprompt__(instance_prompt, self.dict_prompts[instance_target.lower()+"_near"])
            instance_prompt = self.__getprompt__(instance_prompt, instance_target)

        elif self.anchor_type == "absurd":
            instance_anchor_prompt = self.dict_prompts[instance_target.lower()+"_absurd"]#""#instance_prompt
            instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
       
        instance_image, mask = self.preprocess(
            instance_image, random_scale, self.interpolation
        )

        if random_scale < 0.6 * self.size:
            instance_prompt = (
                np.random.choice(["a far away ", "very small "]) + instance_prompt
            )
        elif random_scale > self.size:
            instance_prompt = (
                np.random.choice(["zoomed in ", "close up "]) + instance_prompt
            )

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[
                index % self.num_class_images
            ]
           
            class_image = Image.open(class_image)
           
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def isimage(path):
    if "png" in path.lower() or "jpg" in path.lower() or "jpeg" in path.lower():
        return True


def filter(
    folder,
    impath,
    outpath=None,
    unfiltered_path=None,
    threshold=0.15,
    image_threshold=0.5,
    anchor_size=10,
    target_size=3,
    return_score=False,
):
    model = torch.jit.load(
        "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt"
    )
    if isinstance(folder, list):
        image_paths = folder
        image_captions = ["None" for _ in range(len(image_paths))]
    elif Path(folder / "images.txt").exists():
        with open(f"{folder}/images.txt", "r") as f:
            image_paths = f.read().splitlines()
        with open(f"{folder}/caption.txt", "r") as f:
            image_captions = f.read().splitlines()
    else:
        image_paths = [
            os.path.join(str(folder), file_path)
            for file_path in os.listdir(folder)
            if isimage(file_path)
        ]
        image_captions = ["None" for _ in range(len(image_paths))]

    batch = small_288(Image.open(impath).convert("RGB")).unsqueeze(0)
    embedding_target = model(batch)[0, :]

    filtered_paths = []
    filtered_captions = []
    unfiltered_paths = []
    unfiltered_captions = []
    count_dict = {}
    for im, c in zip(image_paths, image_captions):
        if c not in count_dict:
            count_dict[c] = 0
        if isinstance(folder, list):
            batch = small_288(im).unsqueeze(0)
        else:
            batch = small_288(Image.open(im).convert("RGB")).unsqueeze(0)
        embedding = model(batch)[0, :]

        diff_sscd = (embedding * embedding_target).sum()

        if diff_sscd <= image_threshold:
            filtered_paths.append(im)
            filtered_captions.append(c)
            count_dict[c] += 1
        else:
            unfiltered_paths.append(im)
            unfiltered_captions.append(c)

    # only return score
    if return_score:
        score = len(unfiltered_paths) / (len(unfiltered_paths) + len(filtered_paths))
        return score

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/samples", exist_ok=True)
    with open(f"{outpath}/caption.txt", "w") as f:
        for each in filtered_captions:
            f.write(each.strip() + "\n")

    with open(f"{outpath}/images.txt", "w") as f:
        for each in filtered_paths:
            f.write(each.strip() + "\n")
            imbase = Path(each).name
            shutil.copy(each, f"{outpath}/samples/{imbase}")

    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("+ Filter Summary +")
    print(f"+ Remained images: {len(filtered_paths)}")
    print(f"+ Filtered images: {len(unfiltered_paths)}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")

    sorted_list = sorted(list(count_dict.items()), key=lambda x: x[1], reverse=True)
    anchor_prompts = [c[0] for c in sorted_list[:anchor_size]]
    target_prompts = [c[0] for c in sorted_list[-target_size:]]
    return anchor_prompts, target_prompts, len(filtered_paths)


def getanchorprompts(
    pipeline,
    accelerator,
    class_prompt,
    concept_type,
    class_images_dir,
    num_class_images=200,
    mem_impath=None,
    model_id="meta-llama",
):
    if model_id == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    class_prompt_collection = []
    caption_target = []
    if concept_type in ["object", "nudity", "inappropriate_content"]:
        if model_id == "openai":
            messages = [
                {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
                {"role": "user", "content": f'Generate {num_class_images} captions for images containing a {class_prompt}. The caption should also contain the word "{class_prompt}" '},
            ]
        else:
            messages = [
                    {"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."},
                    {"role": "user", "content": f'''Generate {num_class_images} caption for images containing a {class_prompt}. The caption should also contain the word "{class_prompt}". DO NOT add any unnecessary adjectives or emotion words in the caption. Please keep the caption factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation.

                            Example captions for the category "cat" are:
                            1. A photo of a siamese cat playing in a garden.
                            2. A cat is sitting beside a book in a library.
                            4. Watercolor style painting of a cat. '''
                    }, ]
        numtries = 0
        while True:
            if model_id == "openai":
                outputs = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=messages
                ).choices[0].message.content.lower().split("\n")
            else:
                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                outputs = model(
                    messages,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )[0]["generated_text"][-1]['content'].split("\n")[1:-1]

            print(outputs)
            if concept_type in ["object", "nudity", "inappropriate_content"]:
                class_prompt_collection += [
                    x for x in outputs if x != ''
                ]
            else:
                class_prompt_collection += [
                    x
                    for x in outputs
                    if (class_prompt in x and x != '')
                ]
            messages.append(
                {"role": "assistant", "content": outputs}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Generate {num_class_images-len(class_prompt_collection)} more captions",
                }
            )
            messages = messages[min(len(messages),-10):]
            print(len(class_prompt_collection))
            numtries +=1
            if len(class_prompt_collection) >= num_class_images or numtries > 10:
                break
        class_prompt_collection = clean_prompt(class_prompt_collection)[
            :num_class_images
        ]

    elif concept_type == "memorization":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        num_prompts_firstpass = 5
        num_prompts_secondpass = 2
        threshold = 0.3

        os.makedirs(class_images_dir / "temp/", exist_ok=True)
        class_prompt_collection_counter = []
        caption_target = []
        prev_captions = []
        messages = [
            {
                "role": "user",
                "content": f"Generate {4*num_prompts_firstpass} different paraphrase of the caption: {class_prompt}. Preserve the meaning when paraphrasing.",
            }
        ]
        while True:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

            class_prompt_collection_ = [
                x.strip()
                for x in completion.choices[0].message.content.lower().split("\n")
                if x.strip() != ""
            ]
            class_prompt_collection_ = clean_prompt(class_prompt_collection_)

            for prompt in tqdm(
                class_prompt_collection_,
                desc="Generating anchor and target prompts ",
                disable=not accelerator.is_local_main_process,
            ):
                print(f"Prompt: {prompt}")
                images = pipeline(
                    [prompt] * 10,
                    num_inference_steps=25,
                ).images

                score = filter(images, mem_impath, return_score=True)
                print(f"Memorization rate: {score}")
                if (
                    score <= threshold
                    and prompt not in class_prompt_collection
                    and len(class_prompt_collection) < num_prompts_firstpass
                ):
                    class_prompt_collection += [prompt]
                    class_prompt_collection_counter += [score]
                elif (
                    score >= 0.6
                    and prompt not in caption_target
                    and len(caption_target) < 2
                ):
                    caption_target += [prompt]
                if (
                    len(class_prompt_collection) >= num_prompts_firstpass
                    and len(caption_target) >= 2
                ):
                    break

            if len(class_prompt_collection) >= num_prompts_firstpass:
                break
            prev_captions += class_prompt_collection_
            prev_captions_ = ",".join(prev_captions[-40:])

            messages = [
                {
                    "role": "user",
                    "content": f"Generate {4*(num_prompts_firstpass- len(class_prompt_collection))} different paraphrase of the caption: {class_prompt}. Preserve the meaning the most when paraphrasing. Also make sure that the new captions are different from the following captions: {prev_captions_[:4000]}",
                }
            ]


        for prompt in class_prompt_collection[:num_prompts_firstpass]:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate {num_prompts_secondpass} different paraphrases of: {prompt}. ",
                    }
                ],
            )
            class_prompt_collection += clean_prompt(
                [
                    x.strip()
                    for x in completion.choices[0].message.content.lower().split("\n")
                    if x.strip() != ""
                ]
            )

        for prompt in tqdm(
            class_prompt_collection[num_prompts_firstpass:],
            desc="Memorization rate for final prompts",
        ):
            images = pipeline(
                [prompt] * 10,
                num_inference_steps=25,
            ).images

            class_prompt_collection_counter += [
                filter(images, mem_impath, return_score=True)
            ]


        class_prompt_collection = sorted(
            zip(class_prompt_collection, class_prompt_collection_counter),
            key=lambda x: x[1],
        )
        caption_target += [x for (x, y) in class_prompt_collection if y >= 0.6]
        class_prompt_collection = [
            x for (x, y) in class_prompt_collection if y <= threshold
        ][:10]
        print("Anchor prompts:", class_prompt_collection)
        print("Target prompts:", caption_target)
    return class_prompt_collection, ";*+".join(caption_target)


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in class_prompt_collection
    ]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    class_prompt_collection = [x.replace('"', "") for x in class_prompt_collection]
    return class_prompt_collection


def safe_dir(dir):
    if not dir.exists():
        os.makedirs(str(dir), exist_ok=True)
    return dir


import argparse


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=["style", "object", "memorization", "nudity", "inappropriate_content"],
        help="the type of removed concepts",
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        required=True,
        help="target style to remove, used when kldiv loss",
    )
    parser.add_argument(
        "--prompt_gen_model",
        type=str,
        default="meta-llama",
        choices=["openai", "meta-llama"],
        help="the type of model to generate anchor prompts",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--mem_impath",
        type=str,
        default="",
        help="the path to saved memorized image. Required when concept_type is memorization",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--use_current_model_for_anchor",
        default=False,
        action="store_true",
        help="Flag to use the current model to update the unlearned model.",
    )
    
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=.1,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="the number of generated images used for ablating the concept",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=1000,
        help=(
            "Minimal anchor class images. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--num_class_prompts",
        type=int,
        default=200,
        help=("Minimal prompts used to generate anchor class images"),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--importance_sampling",
        action="store_true",
        default=False,
        help="use lower t for finetuning, as MACE paper",
    )
    
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--f_divergence_type",
        type=str,
          default="mse",
        choices=["mse",'kl', 'reverse_kl', 'hellinger', 'jensen_shannon', 'pearson_chi2','total_variation'],
        help="select the type of f*",
    )
    parser.add_argument(
        "--anchor_type",
        type=str,
          default="superclass",
        choices=['superclass', 'empty', 'absurd','near'],
        help="select the type of anchor choice",
    )
    parser.add_argument(
        "--variational",
        action="store_true",
        help=(
            "USE VARIATIONAL"
        ),
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default="cross-attn",
        choices=["full-weight", "attn", "cross-attn", "embedding"],
        help="parameter groups to finetune. Default: full-weight for memorization and cross-attn for others",
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default="model-based",
        help="loss type for reverse fine-tuning",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant", # "constant"
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--clip_gradients",
        action="store_true",
        help="Whether or not to clip the gradients",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--hflip", action="store_true", help="Apply horizontal flip data augmentation."
    )
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


def parse_args_flux(input_args=None):
    parser = argparse.ArgumentParser(description="FLUX Concept Erasure Training Script.")
    
    # --- Model and Revision ---
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained FLUX model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    # --- Concept Erasure Specifics ---
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=["style", "object", "memorization", "nudity", "inappropriate_content"],
        help="The type of concept being erased.",
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        required=True,
        help="Target style or concept name to remove.",
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default="context-text-embed",
        choices=["context-text-embed", "context_embedder", "text_embedder", "full-weight"],
        help="Transformer layers to fine-tune. FLUX targets context/text embedders for erasure.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for the T5-XXL text encoder (default: 512 for FLUX).",
    )
    parser.add_argument(
        "--clip_gradients",
        action="store_true",
        help="Whether or not to clip the gradients",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=1000,
        help=(
            "Minimal anchor class images. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    # --- Data and Prompts ---
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        help="The prompt used to specify the instance/concept.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify anchor/class images.",
    )
    parser.add_argument(
        "--prompt_gen_model",
        type=str,
        default="meta-llama",
        choices=["openai", "meta-llama"],
        help="Model used to generate anchor prompts.",
    )
    parser.add_argument(
        "--mem_impath",
        type=str,
        default="",
        help="Path to saved memorized image (required for memorization erasure).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # --- Training Hyperparameters ---
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop images.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size for sampling anchor images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps (overrides epochs).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale learning rate by batch size and GPUs.",
    )
    
    # --- Loss and Optimization ---
    parser.add_argument(
        "--f_divergence_type",
        type=str,
        default="mse",
        choices=["mse", "hellinger", "pearson_chi2"],
        help="Divergence loss type for erasure.",
    )
    parser.add_argument(
        "--anchor_type",
        type=str,
        default="superclass",
        choices=['superclass', 'empty', 'absurd', 'near'],
        help="Type of anchor/class prompt choice.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation (class) loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=0.1,
        help="Weight of the prior preservation loss.",
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default="model-based",
        help="Target for erasure: model-based (frozen model) or noise-based.",
    )
    parser.add_argument(
        "--use_current_model_for_anchor",
        default=False,
        action="store_true",
        help="Use current weights (not frozen weights) for the anchor prediction.",
    )
    parser.add_argument(
        "--importance_sampling",
        action="store_true",
        default=False,
        help="Focus training on specific timesteps (as per MACE).",
    )
    
    # --- Scheduler and Optimizers ---
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Scheduler type: ["linear", "cosine", "polynomial", "constant"].',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Steps for lr warmup.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    
    # --- Checkpointing and Validation ---
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt to test erasure during training.",
    )
    parser.add_argument("--num_validation_images", type=int, default=2)
    parser.add_argument("--validation_steps", type=int, default=500)
    
    # --- Efficiency and Logging ---
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-erasure-model",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    # --- Infrastructure ---
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to JSON containing multiple concepts.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--noaug", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # --- Final Arg Processing ---
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")

    return args


class GradientTracker:
    def __init__(self):
        self.gradient_stats = defaultdict(list)

    def compute_gradient_stats(self, params):
        gradient_norms = []
        gradient_norms_std = []
        gradient_norms_max = []

        for param in params:
            if param.grad is not None:
               
                grad_norm = param.grad.data.norm(1).item()
                gradient_norms.append(grad_norm)
        if gradient_norms:
            mean_grad = np.mean(gradient_norms)
            std_grad = np.std(gradient_norms)
            return mean_grad, std_grad, gradient_norms
        else:
            return 0.0, 0.0, []

    def track_gradients(self, model, epoch):
        """Track gradients for a specific epoch"""
        mean_grad, std_grad, grad_norms = self.compute_gradient_stats(model)

        self.gradient_stats['epoch'].append(epoch)
        self.gradient_stats['mean_gradient'].append(mean_grad)
        self.gradient_stats['std_gradient'].append(std_grad)
        self.gradient_stats['all_gradients'].append(grad_norms)

        return mean_grad, std_grad



class DetailedGradientTracker:
    def __init__(self):
        self.layer_gradient_stats = defaultdict(lambda: defaultdict(list))

    def track_layer_gradients(self, model, epoch):
        """Track gradients per layer"""
        overall_gradients = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                overall_gradients.append(grad_norm)

                # Store per-layer statistics
                self.layer_gradient_stats[name]['epoch'].append(epoch)
                self.layer_gradient_stats[name]['gradient_norm'].append(grad_norm)

        # Overall statistics
        if overall_gradients:
            mean_grad = np.mean(overall_gradients)
            std_grad = np.std(overall_gradients)

            self.layer_gradient_stats['overall']['epoch'].append(epoch)
            self.layer_gradient_stats['overall']['mean_gradient'].append(mean_grad)
            self.layer_gradient_stats['overall']['std_gradient'].append(std_grad)

            return mean_grad, std_grad
        return 0.0, 0.0

# Save gradients to file
def save_gradient_stats(gradient_tracker, filename='gradient_stats.npz'):
    """Save gradient statistics to file"""
    np.savez(filename, **dict(gradient_tracker.gradient_stats))
    print(f"Gradient statistics saved to {filename}")

# Load gradients from file
def load_gradient_stats(filename='gradient_stats.npz'):
    """Load gradient statistics from file"""
    data = np.load(filename, allow_pickle=True)
    gradient_stats = {}
    for key in data.files:
        gradient_stats[key] = data[key]
    return gradient_stats

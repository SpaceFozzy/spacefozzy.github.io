---
title: "Illustrating Literature with Open Source Deep Learning Models"
date: 2025-01-09 17:49:30 +300
categories: [Applied Gen AI]
tags: [diffusers, transformers, literature, python]     # TAG names should always be lowercase
---
A guide to illustrating stories with generative AI
---
With some Python and a CUDA-enabled GPU, one person can now illustrate their favourite public domain lit with remarkable visual density. They will have to manage certain inconsistencies in characters and setting (a limitation of the current technology), but with recent advances in transformer and diffusion models the capacity for one person to create digital media is rapidly increasing. 

![img-description](/assets/img/22.png){: width="640" height="640" }
_"there was much breathless talk of new elements, bizarre optical properties, and other things which puzzled men of science are wont to say when faced by the unknown"_

The opportunity here at the intersection of deep learning and art is broadly felt. There are many projects currently under active development applying language models to interactive fiction while character consistency remains a coveted achievement in recent research  and diffusion communities. Yet the application of machine learning to the humanistic domain of art is a highly-charged subject. We've been blindsided by algorithms that can illustrate and write before fully automating the mundane parts of our lives, with the outrageous twist that their fuel is our own unattributed creative output.

Yet, with that acknowledged, a proliferation of art where none would have otherwise been would be, in my opinion, an undeniable good. To that end, this article is a guide to illustrating stories with digital media in a way that wasn't technically possible five years ago.

## The Blueprint
The process looks something like this:
* Acquire access to a graphics card (GPU) and Python
* Break the story into 100-word chunks
* Use a  language model to extract a text description of a scene representing the chunk
* Use that scene as a prompt for a text-to-image diffusion model to generate images 
* Build the interleaved story and images into an easily-scrollable static website for a modern audience.

![img-description](/assets/img/30.png){: width="640" height="640" }
_"they had indeed seen with waking eyes that cryptic vestige of the fathomless gulfs outside; that lone, weird message from other universes and other realms of matter, force, and entity"_


## A Walk-through: HP Lovecraft's "The Colour Out of Space"

I chose H.P. Lovecraft's The Colour Out of Space for my personal example. You can [view the result here](https://spacefozzy.github.io/graphic-novelator/example/pages/1.html). At 12,400 words, the story spans 25 interlinked pages with 5 illustrations each, breaking up the 100 word chunks of the short story nicely. The [full story text](https://www.gutenberg.org/ebooks/68236) was available on Project Gutenberg as a mere 89kb plain text file and, like the others there, is in the public domain. 

A single pass took about 1.5 hours to generate locally on my economic NVIDIA RTX 3060 GPU . To get the example visuals to the state where I thought it was shareable, I did three full passes, cherry-picked the best generations, then recreated individual images or scene descriptions as needed to get the particularly "stubborn" scenes where I wanted them. All said, it still took about a day to generate and curate the images for the project, and while they aren't perfect, I'm encouraged by the overall result.

![img-description](/assets/img/100.png){: width="640" height="640" }
_"a thousand tiny points of faint and unhallowed radiance, tipping each bough like the fire of St. Elmo"_

## Accessing a GPU
Aiming for bite-sized chunks of 100 words does mean a lot of images. A standard 1024 x 1024 DALL-E 3 image costs $0.04 USD from OpenAI at the time of this writing. For the example short story I generated 375+ images (125 images x 3 passes, plus extras where needed). That would be $15+ USD for my example story, all without the freedom to adjust the model.

The text and image generation models we need are both deep learning models that require a graphics card (GPU) to perform costly matrix multiplication in decent time.  Google offers a free service called Google Colab where you can access Python notebooks and GPUs for free (though you must pay for priority access). You could run all this in a [Google Colab notebook](https://colab.research.google.com/), [connecting to Google Drive](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=Google_Drive) to read the story text and write the images.

However, I opted to get a GPU and run both models on it locally. After some impatient research, I spent $450 CDN on an NVIDIA RTX 3060 with 12GB of VRAM. The card is past its prime, but still well-known because the 12GB is higher than most in its price bracket. All that memory is useful for fitting as much of a model as possible on a card.

![img-description](/assets/img/3060.jpg){: width="640" height="480" }
_The GPU I chose to start running Llama3 8b Instruct and Stable Diffusion XL locally_

It turns out this was a great purchase for what I wanted. It runs Llama 3 8b Instruct (quantized) and Stable Diffusion XL all fast enough for me. Illustrating a single chunk takes 30s-60s.

![img-description](/assets/img/3060-installed.jpg){: width="640" height="480" }
_A happily installed GPU._

## The Intuition
At a high level, my approach to the process works like this: an arbitrary text file is loaded into memory and is broken up into chunks of 100 words, all in a Python list.

![img-description](/assets/img/intuition-1.png){: width="550" height="207" }
_The story text is broken into chunks of 100 words._

The story text is broken into chunks of 100 words.Each `Chunk` tracks its own story text, scene description and image using the local file system. They also expose functions to generate any of those properties that are missing.
Each `Chunk` of the story tracks its text, scene description, and image.

![img-description](/assets/img/intuition-2.png){: width="550" height="294" }
_Each Chunk of the story tracks its text, scene description, and image._

## The Models
One of the advantages of running local models is selecting them according to your preference. I used [Hugging Face](https://huggingface.co/) (a renowned model hub and Python library) to download and run inference on the models.

The three models I ultimately selected for the project were
* [Meta-Llama-3–8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
* [Stable Diffusion XL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
* [Stable Diffusion XL Refiner 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

Their pages have instructions for invoking the models. As a gated model, Llama3 also requires that you accept their Terms and Conditions before you can access it. I also didn't realize when I started that Stable Diffusion XL uniquely comes in two parts (base and refiner) - but it does, and the project uses both.

Since the time of writing this, new models in both the Stable Diffusion and Llama families have been released, and they contend with many others in their respective domains. This project could easily be done with, say, Microsoft's [Phi language models](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3), or Black Forest Lab's [Flux image model](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3). Hugging Face supports them both. Not only that, to hone in on a specific image style you could [add what's called a "LoRA"](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference) or even [train your own](https://huggingface.co/docs/diffusers/en/training/lora). 

For the purposes of this article, Llama 3 and SDXL will do just fine.

![img-description](/assets/img/17.png){: width="640" height="640" }
_"not feared half so much as the small island in the Miskatonic where the devil held court beside a curious stone altar"_

## The Code

This is all coordinated by a main script which processes the story in the following way.

1. [Generate all the chunks](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L41) from the text file
2. [Load Llama3](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L38) onto the GPU
3. Iterate through the chunks and [generate their scene descriptions](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L49)
4. [Unload Llama3](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L55) and [load SDXL](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L57) onto the GPU
5. Iterate through the chunks and [generate their images](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L57)
6. Pass the chunks to an instance of `HtmlBuilder` to generate [all the web pages for the story](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L65), along with [an index](https://github.com/SpaceFozzy/graphic-novelator/blob/d3f4e3ef3fcd20e3b7a01ceb196e6e178be8b538/app/start.py#L66)

And there are three environment variables for configuration:

* `AUTHOR` is the author's name
* `TITLE` is the story's title
* `STORY_DIR` is the path where the story text file is located, and where the story will be generated, e.g. `"../the-festival"`

An example run looks like this:
```shell
TITLE="The Festival" \
AUTHOR="H.P. Lovecraft" \
STORY_DIR="../the-festiva-illustrated" \
python3 app/start.py
```

(view the output for "The Festival" [here](https://spacefozzy.github.io/the-festival-illustrated/index.html))

And finally, you can pass a chunk number to force a chunk (and only that chunk) to be regenerated. i.e. if you notice the image for chunk 7 doesn't work, you can run `python3 app/start.py 7` to regenerate both the description and image.

![img-description](/assets/img/96.png){: width="640" height="640" }
_"unknown and unholy iridescence from the slimy depths in front"_

## Limitations

### Character Consistency

A well-known shortcoming of the current state of image generation is the struggle for consistency in characters between images (and now video). It's a  stubborn problem that is currently a focus among AI content creation communities. While diffusion models can be fined-tuned on faces and characters, the fine details of clothing are not so easily controlled from one image to the next. Consistent faces are relatively achievable, but collars disappear, buttons re-align, and pockets run amok, requiring the attention of a capable digital artist to bring them into harmony. The progress here has been interesting to follow:

* [Textual Inversions](https://arxiv.org/pdf/2208.01618), given a few images, will find an embedding that represents them and provide you with a new token for that embedding to be used in your prompts, evoking that concept in future generations.

* [Low Rank Adaptations (LoRA)](https://arxiv.org/pdf/2106.09685) is a type of performance efficient fine-tuning that allows you to fine-tune a lower-dimensional layer on top of an existing model with far fewer resources than training all the parameters (i.e. on a consumer graphics card). Countless character, style, and object-specific LoRAs exist out there in the wild for you to discover.

* [IP Adapter](https://arxiv.org/pdf/2308.06721) modifies the diffusion model's cross-attention to take an image prompt in addition to a text prompt. Many online guides show how to combine IP Adapter with masking and ControlNet to consistently render outfits on models (mostly for fashion applications).


![img-description](/assets/img/63.png){: width="640" height="640" }
_"Three days later Nahum burst into Ammi's kitchen in the early morning, and in the absence of his host stammered out a desperate tale once more"_
Innovation in this area is active. A couple new approaches I've recently tried: 

1. [The Chosen One: Consistent Characters in Text-to-Image Diffusion Models](https://arxiv.org/pdf/2311.10093) iteratively fine-tunes a model on the most consistent grouping of generations for a subject, "funnelling" its internal representation into a more consistent identity.

2. [StoryMaker: Towards Holistic Consistent Characters In Text-To-Image Generation](https://arxiv.org/pdf/2409.12576) builds on IP Adapter to include image prompting specifically for a character's face and outfit. 

In my experience fine-tuning  results in small but noticeable inconsistencies, while IP Adapter techniques are better but at the cost of flexibility in angles and framing.

### Story Consistency

Unless your story fits entirely within the context window of the language model extracting the scene descriptions, the story must be chunked into pages to be illustrated. However, at the time the scene descriptions are extracted, the model is blind to anything beyond the current chunk. So, a character who entered the room in a previous chunk, but is not mentioned in the current chunk, would not be captured in the resulting illustration. This can give rise to various continuity errors.

Similarly, story elements may be referenced in a way where more context is needed beyond the current chunk to understand exactly what is going on.  For example, finding the phrase "… as he transformed into a bat!" at the beginning of a chunk would leave the language model unable to know who exactly underwent the transformation. This is known as "coreference resolution" in natural language processing, and is one of the challenges in rendering stories in this chunked, linear fashion.

As a concrete example, I had a silly time attempting Mary Shelley's Frankenstein, where the antagonist is often just referred to as the creature. This sent me down a rabbit hole of considering how to resolve these references, but also how to track character details (and their evolutions) as a story progresses just for more accurate images.

### Detail Accuracy
Faces, hands and "fidgety little things" (like pouches on belts) can be notoriously mangled by image models. There are specific "detailers" that will identify faces, redraw them enlarged, then scale them back down and stitch them into the original image with surprisingly good results. However, those workflows come with a more complex configuration. ComfyUI and its thriving community support a whole universe of workflows for more control over the images you generate. Model selection can also make a big difference here. "Inpainting" is also a technique that allows to you regenerate specific parts of an image, but resists automation. Though, with the current state of things, you should expect to be  cherry-pick the best images, or spending some time touching them up.

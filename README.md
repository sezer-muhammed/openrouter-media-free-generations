# openrouter-media-free-generations

A small Python library for generating images through OpenRouter and discovering the current image models. It uses the [dedicated OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).

**The package is free to use; OpenRouter image generation is generally billed by the selected model and provider.** Check live pricing in `ImageModelsList` before generating. This project is independent of OpenRouter.

## Install

```bash
python -m pip install -e .
```

Copy `.env.example` to `.env` and put your key there. `.env` is ignored by Git. The library reads `OPENROUTER_API_KEY` from the process environment, or you can pass `api_token` directly. For a local shell session:

```bash
set -a
source .env
set +a
```

## Generate and save

```python
from openrouter_media_free_generations import ImageGenerator

generator = ImageGenerator(
    "bytedance-seed/seedream-4.5",
    api_token=None,  # Uses OPENROUTER_API_KEY when omitted
    resolution="1K",
    aspect_ratio="1:1",
)

result = generator.generate("A red panda astronaut floating in space")
print(result.usage)             # Includes cost when OpenRouter reports it
path = generator.save("generated/red-panda")  # Adds the image extension
print(path)

# Per-call options override the constructor defaults:
# generator.generate("A mountain at sunrise", aspect_ratio="16:9")
```

`generate()` returns a `GeneratedImages` object. Its `images` tuple contains image bytes and media types. For multiple outputs, request `n` if the model supports it, then call `result.save("second", index=1)`. You can also call `result.images[0].save("first")`.

Model-specific parameters are passed through to OpenRouter. Consult each model's `supported_parameters`; provider-specific settings can be passed using `provider={"options": {"provider-slug": {...}}}`. For endpoints with native SSE, call `generator.generate(prompt, stream=True, on_event=callback)`. The callback receives event metadata without large base64 image data.

## Interactive example

```bash
python -m pip install -e '.[demo]'
python examples/interactive_generate.py
```

The example reads the ignored `.env` file automatically. It lists **all live image models**, showing input/output modalities and each provider's published pricing lines. Routes are sorted from low to high by their lowest published output rate. Rates may be per image, token, or megapixel, so this ordering is not an estimated total generation cost. A `FREE` label means **every published pricing line is zero**; it does not depend on a model name or API label.

Enter a model number and prompt. The example then asks for supported parameters with their allowed values or ranges. When `webp` is supported, leaving `output_format` blank selects WebP. Otherwise it uses the provider default. If the chosen model needs a reference image, enter a local image path or URL. Images go into `generated/` with the correct extension.

The chosen provider is pinned and fallback is disabled so pricing and capabilities match the displayed route. `tqdm` shows elapsed seconds; on SSE capable endpoints it also counts partial image events. OpenRouter does not publish a completion percentage for this operation, so the indicator cannot show an accurate percent. Currently, zero-price endpoints do not support SSE.

## Fetch current image models

```python
from openrouter_media_free_generations import ImageModelsList

models = ImageModelsList()
data = models.fetch()             # Includes per-provider capabilities and pricing
print(len(data["data"]))
print(data["data"][0])

ImageModelsList().save_json("generated/image-models.json")
```

`fetch()` returns OpenRouter's model JSON with an `endpoint_details` field added to each model. The default makes one request per model to include provider details and pricing. Use `fetch(include_endpoints=False)` for the faster model summary only. No model list or prices are hardcoded.

## Run tests

```bash
python -m unittest discover -s tests -v
```

Tests mock image generation; they do not spend OpenRouter credits. An actual generation needs your key and available credits.

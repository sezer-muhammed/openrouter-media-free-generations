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

Model-specific parameters are passed through to OpenRouter. Consult each model's `supported_parameters`; provider-specific settings can be passed using `provider={"options": {"provider-slug": {...}}}`. Streaming is outside this first version.

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

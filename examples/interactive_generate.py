"""Interactive OpenRouter image model picker. Run: python examples/interactive_generate.py"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

from openrouter_media_free_generations import ImageGenerator, ImageModelsList, OpenRouterError


ROOT = Path(__file__).resolve().parents[1]


def load_local_env() -> None:
    """Read only the API key from the repository's ignored .env file."""
    env_file = ROOT / ".env"
    if not env_file.exists() or os.environ.get("OPENROUTER_API_KEY"):
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENROUTER_API_KEY":
            os.environ["OPENROUTER_API_KEY"] = value.strip().strip('"\'')
            return


def is_free(endpoint: dict) -> bool:
    prices = endpoint.get("pricing")
    if not prices:
        return False
    try:
        return all(float(line["cost_usd"]) == 0 for line in prices)
    except (KeyError, TypeError, ValueError):
        return False


def output_price(endpoint: dict) -> float:
    """Lowest published output rate; units are displayed because they differ."""
    prices = endpoint.get("pricing") or []
    values = []
    for line in prices:
        if str(line.get("billable", "")).startswith("output_"):
            try:
                values.append(float(line["cost_usd"]))
            except (KeyError, TypeError, ValueError):
                pass
    return min(values, default=float("inf"))


def choices(models: list[dict]) -> list[tuple[dict, dict | None]]:
    result = []
    for model in models:
        endpoints = model.get("endpoint_details", {}).get("endpoints", [])
        if endpoints:
            result.extend((model, endpoint) for endpoint in endpoints)
        else:
            result.append((model, None))
    return sorted(result, key=lambda pair: (output_price(pair[1] or {}), pair[0]["id"], (pair[1] or {}).get("provider_tag") or ""))


def format_prices(endpoint: dict) -> str:
    prices = endpoint.get("pricing") or []
    if not prices:
        return "price unavailable"
    return "; ".join(
        f"{line.get('billable', '?')} ${float(line['cost_usd']):g}/{line.get('unit', '?')}"
        + (f" ({line['variant']})" if line.get("variant") else "")
        for line in prices
    )


def ask_selection(options: list[tuple[dict, dict | None]]) -> tuple[dict, dict | None]:
    print("\nImage generation models (live OpenRouter data):")
    print("Sorted by lowest published output rate. Units differ; this is not an estimated total generation cost.\n")
    for number, (model, endpoint) in enumerate(options, 1):
        route = endpoint or {}
        architecture = model.get("architecture") or {}
        input_types = ",".join(architecture.get("input_modalities") or []) or "?"
        output_types = ",".join(architecture.get("output_modalities") or []) or "?"
        provider = route.get("provider_tag") or "no endpoint"
        label = "FREE | " if is_free(route) else ""
        streaming = "SSE" if route.get("supports_streaming") else "no SSE"
        print(f"{number:>2}. {model['id']} [{provider}] | input: {input_types} → output: {output_types} | {label}{format_prices(route)} | {streaming}")
    while True:
        selection = input("\nModel number (q to quit): ").strip().lower()
        if selection in {"q", "quit"}:
            raise KeyboardInterrupt
        if selection.isdigit() and 1 <= int(selection) <= len(options):
            return options[int(selection) - 1]
        print(f"Enter a number from 1 to {len(options)}.")


def reference_from_input(value: str) -> dict:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        url = value
    else:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError("File not found. Enter a local file path or HTTP(S) URL.")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url}}


def ask_references(descriptor: dict) -> list[dict]:
    minimum = int(descriptor.get("min", 0))
    maximum = int(descriptor.get("max", 1))
    print(f"Reference images: {minimum}–{maximum}. Enter a local file path or HTTP(S) URL.")
    references = []
    while len(references) < maximum:
        required = len(references) < minimum
        entry = input(f"Reference {len(references) + 1}{' (required)' if required else ' (blank to skip)'}: ").strip()
        if not entry and not required:
            break
        try:
            references.append(reference_from_input(entry))
        except ValueError as exc:
            print(exc)
    return references


def ask_parameters(endpoint: dict) -> dict:
    result = {}
    descriptors = endpoint.get("supported_parameters", {})
    for name, descriptor in descriptors.items():
        if name in {"stream", "input_references"}:
            continue
        kind = descriptor.get("type")
        if kind == "enum":
            values = descriptor.get("values", [])
            hint = "choices: " + ", ".join(map(str, values))
        elif kind == "range":
            hint = f"integer range: {descriptor.get('min')}–{descriptor.get('max')}"
        else:
            hint = "JSON value, e.g. 42 or true"
        default = "webp" if name == "output_format" and "webp" in descriptor.get("values", []) else None
        while True:
            raw = input(f"{name} ({hint}; blank={'webp' if default else 'provider default'}): ").strip()
            if not raw:
                if default:
                    result[name] = default
                break
            try:
                if kind == "enum":
                    if raw not in map(str, values):
                        raise ValueError("Choose one of the listed values.")
                    result[name] = raw
                elif kind == "range":
                    value = int(raw)
                    if not int(descriptor["min"]) <= value <= int(descriptor["max"]):
                        raise ValueError("Value is outside the range.")
                    result[name] = value
                else:
                    result[name] = json.loads(raw)
                break
            except (ValueError, KeyError) as exc:
                print(f"Invalid value: {exc}")
    refs = descriptors.get("input_references")
    if refs and int(refs.get("max", 0)) > 0:
        selected = ask_references(refs)
        if selected:
            result["input_references"] = selected
    return result


def generate_with_progress(generator: ImageGenerator, prompt: str, parameters: dict, streaming: bool):
    stop = threading.Event()
    partials = 0
    with tqdm(total=None, unit="sec", desc="Generating", dynamic_ncols=True) as progress:
        def tick() -> None:
            while not stop.wait(1):
                progress.update(1)

        def on_event(event: dict) -> None:
            nonlocal partials
            if event.get("type") == "image_generation.partial_image":
                partials += 1
                progress.set_postfix_str(f"partial images: {partials}")

        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()
        try:
            return generator.generate(prompt, stream=True, on_event=on_event, **parameters) if streaming else generator.generate(prompt, **parameters)
        finally:
            stop.set()
            ticker.join(timeout=2)


def main() -> int:
    load_local_env()
    try:
        models = ImageModelsList().fetch()["data"]
        options = choices(models)
        if not options:
            print("No image generation models are currently available.")
            return 1
        model, endpoint = ask_selection(options)
        prompt = input("Prompt: ").strip()
        if not prompt:
            print("Prompt cannot be empty.")
            return 1
        parameters = ask_parameters(endpoint or {"supported_parameters": model.get("supported_parameters", {})})
        if endpoint and endpoint.get("provider_tag"):
            # Pin the selected provider so pricing and capabilities match the displayed route.
            parameters["provider"] = {"only": [endpoint["provider_tag"]], "allow_fallbacks": False}
        streaming = bool(endpoint and endpoint.get("supports_streaming"))
        if not streaming:
            print("This endpoint does not support SSE; tqdm will show elapsed time.")
        generator = ImageGenerator(model["id"])
        result = generate_with_progress(generator, prompt, parameters, streaming)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for index, _ in enumerate(result.images):
            path = result.save(ROOT / "generated" / f"{model['id'].replace('/', '-')}-{stamp}-{index + 1}", index)
            print(f"Saved: {path}")
        if "cost" in result.usage:
            print(f"Actual cost: ${result.usage['cost']}")
        return 0
    except (OpenRouterError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

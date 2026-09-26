import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from openrouter_media_free_generations import ImageGenerator, ImageModelsList, OpenRouterError


class ClientTests(unittest.TestCase):
    def test_generate_passes_parameters_and_saves_bytes(self):
        image_bytes = b"\x89PNG\r\n\x1a\nexample"
        response = {
            "data": [{"b64_json": base64.b64encode(image_bytes).decode(), "media_type": "image/png"}],
            "usage": {"cost": 0.04},
        }
        with patch("openrouter_media_free_generations.client._request", return_value=response) as request:
            generator = ImageGenerator("example/model", "test-key", aspect_ratio="1:1")
            result = generator.generate("A cat", aspect_ratio="16:9", seed=4)
            request.assert_called_once_with(
                "/api/v1/images", "test-key",
                {"model": "example/model", "prompt": "A cat", "aspect_ratio": "16:9", "seed": 4},
            )
        self.assertEqual(result.usage["cost"], 0.04)
        with tempfile.TemporaryDirectory() as directory:
            path = generator.save(Path(directory) / "cat")
            self.assertEqual(path.suffix, ".png")
            self.assertEqual(path.read_bytes(), image_bytes)
            dotted = generator.save(Path(directory) / "model-v0.1-result")
            self.assertEqual(dotted.suffix, ".png")
            self.assertEqual(dotted.name, "model-v0.1-result.png")

    def test_generate_rejects_empty_or_malformed_images(self):
        generator = ImageGenerator("example/model", "test-key")
        with self.assertRaises(ValueError):
            generator.generate("")
        with self.assertRaises(ValueError):
            generator.save("image.png")
        with patch("openrouter_media_free_generations.client._request", return_value={"data": []}):
            with self.assertRaises(OpenRouterError):
                generator.generate("A cat")
        with patch("openrouter_media_free_generations.client._request", return_value={"data": [{"b64_json": "!"}]}):
            with self.assertRaises(OpenRouterError):
                generator.generate("A cat")

    def test_model_details_and_json_export(self):
        summary = {"data": [{"id": "example/model", "endpoints": "/api/v1/images/models/example/model/endpoints"}]}
        details = {"id": "example/model", "endpoints": [{"pricing": [{"cost_usd": 0.02}]}]}
        with patch("openrouter_media_free_generations.client._request", side_effect=[summary, details]) as request:
            with tempfile.TemporaryDirectory() as directory:
                path = ImageModelsList("test-key").save_json(Path(directory) / "models.json")
                result = json.loads(path.read_text())
                self.assertEqual(result["data"][0]["endpoint_details"], details)
                self.assertEqual(request.call_count, 2)

    def test_streaming_collects_final_image_and_progress_events(self):
        image_bytes = b"image-data"
        final = {
            "type": "image_generation.completed",
            "b64_json": base64.b64encode(image_bytes).decode(),
            "media_type": "image/webp",
            "usage": {"cost": 0.01},
        }
        stream = (
            'data: {"type":"image_generation.partial_image","partial_image_index":0,"b64_json":"preview"}\n\n'
            + "data: " + json.dumps(final) + "\n\n"
            + "data: [DONE]\n\n"
        ).encode()
        seen = []
        with patch("openrouter_media_free_generations.client.urlopen", return_value=BytesIO(stream)) as open_url:
            result = ImageGenerator("example/model", "test-key").generate("A cat", stream=True, on_event=seen.append)
        self.assertEqual(result.images[0].data, image_bytes)
        self.assertEqual(result.images[0].media_type, "image/webp")
        self.assertEqual(result.usage["cost"], 0.01)
        self.assertEqual([event["type"] for event in seen], ["image_generation.partial_image", "image_generation.completed"])
        self.assertTrue(all("b64_json" not in event for event in seen))
        self.assertTrue(json.loads(open_url.call_args.args[0].data)["stream"])


if __name__ == "__main__":
    unittest.main()

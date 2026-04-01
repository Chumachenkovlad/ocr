"""docTR model loading and inference."""

from doctr.models import ocr_predictor


class DoctrEngine:
    """Thin wrapper around the docTR OCR predictor."""

    def __init__(self, device: str = "cpu") -> None:
        import torch

        use_gpu = device == "gpu" and torch.cuda.is_available()
        self._predictor = ocr_predictor(pretrained=True).to(
            "cuda" if use_gpu else "cpu"
        )

    def ocr(self, doc: list) -> dict:
        """Run OCR on a document and return the exported dict."""
        result = self._predictor(doc)
        return result.export()

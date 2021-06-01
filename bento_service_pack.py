import bentoml
from transformers import AutoModelWithLMHead, AutoTokenizer

from bentoml.adapters import JsonInput

from bentoml.frameworks.transformers import TransformersModelArtifact
from transformer_service import TransformerService
ts = TransformerService()

model_name = "gpt2"
model = AutoModelWithLMHead.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
#>>> # Option 1: Pack using dictionary (recommended)
artifact = {"model": model, "tokenizer": tokenizer}
ts.pack("gptModel", artifact)
saved_path = ts.save()
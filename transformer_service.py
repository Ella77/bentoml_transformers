import bentoml
from transformers import AutoModelWithLMHead, AutoTokenizer

from bentoml.adapters import JsonInput

from bentoml.frameworks.transformers import TransformersModelArtifact

@bentoml.env(pip_packages=["transformers==3.1.0", "torch==1.6.0"])
@bentoml.artifacts([TransformersModelArtifact("gptModel")])
class TransformerService(bentoml.BentoService):
     @bentoml.api(input=JsonInput(), batch=False)

    #  def tokenize(self, inputs: pd.DataFrame):
    #     tokenizer = self.artifacts.tokenizer
    #     if isinstance(inputs, pd.DataFrame):
    #         inputs = inputs.to_numpy()[:, 0].tolist()
    #     else:
    #         inputs = inputs.tolist()  # for predict_clipper
    #     pred_tokens = map(tokenizer.tokenize, inputs)
    #     pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    #     pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
    #     pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
    #     pred_token_ids = tf.constant(list(pred_token_ids), dtype=tf.int32)
    #     return pred_tokecn_ids

     def predict(self, parsed_json):
         #print(parsed_json)
         src_text = parsed_json.get("text")
         model = self.artifacts.gptModel.get("model")
         tokenizer = self.artifacts.gptModel.get("tokenizer")
         input_ids = tokenizer.encode(src_text, return_tensors="pt")
         output = model.generate(input_ids, max_length=50)
         #print(output, output.shape, type(output[0]))
         output = tokenizer.decode(list(output[0]), skip_special_tokens=True)
         return output

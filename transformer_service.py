import bentoml
#from transformers import AutoModelWithLMHead, AutoTokenizer

from bentoml.adapters import JsonInput
from bentoml.types import JSON_CHARSET, JsonSerializable
from bentoml.frameworks.transformers import TransformersModelArtifact
from typing import List

@bentoml.env(pip_packages=["transformers==4.3.3.", "torch==1.6.0"])
@bentoml.artifacts([TransformersModelArtifact("gptModel")])
class TransformerService(bentoml.BentoService):
     @bentoml.api(input=JsonInput(), mb_max_latency=20000, mb_max_batch_size=100, batch=True)

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

     def predict(self, parsed_json_list : List[JsonSerializable]):
         #print(parsed_json_list)
         #print(parsed_json)
         src_text = [parsed_json.get("text") for parsed_json in parsed_json_list]
         print((src_text))
         model = self.artifacts.gptModel.get("model")
         tokenizer = self.artifacts.gptModel.get("tokenizer")
         tokenizer.pad_token = tokenizer.eos_token
         input_ids = tokenizer(src_text, return_tensors="pt",padding=True)
         #print(input_ids)
         outputs = model.generate(input_ids['input_ids'], max_length=10)
         #print(outputs)
         #print(output, output.shape, type(output[0]))
         #output = tokenizer.decode(list(output[0]))
         #for output in outputs.tolist():
         output = tokenizer.batch_decode(outputs,skip_special_tokens=True)
         return output

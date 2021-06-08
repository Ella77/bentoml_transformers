```python bento_service_pack.py```

save the tagname

### first method
```
bentoml containerize TransformerService:$tagname
docker run --rm -p 5000:5000 TransformerService:$tagname --debug --enable-microbatch --gpus device=0 nvidia/cuda
```
### second method
```
bentoml serve TransformerService:latest --port 5000 --enable-microbatch --workers 1
```

```
curl -X POST "http://localhost:5000/predict" -H "accept: */*" -H "Content-Type: application/json" -d "{\"text\":\"Hi\"}
```

## batch branch

```
For client prediction request, it is the same for both batch and non-batch API, the request should contain only one single input item:

curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"text": "best movie ever"}' \
  localhost:5000/predict
```

## docker image
```bento_service_pack.py```

```
@bentoml.env(docker_base_image="pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel")
``` 

 
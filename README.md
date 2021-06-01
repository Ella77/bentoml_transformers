```python bento_service_pack.py```

save the tagname

### first method
```
bentoml containerize TransformerService:$tagname
docker run --rm -p 5000:5000 TransformerService:$tagname —debug —enable-microbatch
```
### second method
```
bentoml serve TransformerService:latest --port 5000 --enable-microbatch --workers 1
```


curl -X POST "http://localhost:5000/predict" -H "accept: */*" -H "Content-Type: application/json" -d "{\"text\":\"Hi\"}

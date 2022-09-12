## H2GCN_PyG

Reimplementation of NeurIPS paper "[Beyond homophily in graph neural networks: Current limitations and effective designs](https://proceedings.neurips.cc/paper/2020/hash/58ae23d878a47004366189884c2f8440-Abstract.html)" based on PyTorch and PyTorch Geometric (PyG).



## Run

```
python main.py
```



## Note

- Besides the original idea of H2GCN, we also try another variant version of it by changing the original GCN-based weighted aggregator to a GCN layer with learnable parameters (labeled as H2GCN-Variant in this project). However, the original H2GCN performs much better than the H2GCN-Variant in this implementation.
- Currently, we just simply fix the number of aggregation hops as 2, and the number of layers is also 2. It may need necessary changes to our codes if the above two parameters change.
## Performance

|Model|Weighter|sequence length|k|batch size|learning rate|ACC|MRR|nDCG@10|AUC|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GraphSage|:x:|32|:x:|800|1e-5|0.702|0.937|0.8411|0.9920|
|GateSage|CNN|32|8|800|1e-5|0.6689|0.7684|0.8212|0.9903|
|GateSage|TFM|32|8|800|1e-5|||||
|GateSage|First|32|8|800|1e-5|0.6342|0.7411|0.8008|0.9889|
|GateSage|BM25|32|8|800|1e-5|||||
|GateSage|Random|32|8|800|1e-5|||||
|GateSage|KeyBert|32|8|800|1e-5|||||

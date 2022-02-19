## Performance

|Model|Weighter|sequence length|k|batch size|learning rate|ACC|MRR|nDCG@10|AUC|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|GraphSage|:x:|32|:x:|800|1e-5|0.702|0.937|0.8411|0.9920|
|GateSage|CNN|32|8|800|1e-5|0.6702|0.7697|0.8223|0.9904|
|GateSage|TFM|32|8|800|1e-5|||||
|GateSage|First|32|8|800|1e-5|0.6342|0.7411|0.8008|0.9889|
|GateSage|BM25|32|8|800|1e-5|0.6558|0.7585|0.8135|0.99|
|GateSage|Random|32|8|800|1e-5|||||
|GateSage|KeyBert|32|8|800|1e-5|||||

- draw each metric line
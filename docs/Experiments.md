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

|k|gate|batch size|ACC|MRR|nDCG|
|:-:|:-:|:-:|:-:|:-:|
|1|First|||
|1|Cnn|||
|2|First|||
|2|Cnn|
|4|First|0.5491|0.6691|0.7423|
|4|Cnn|0.6569|0.7579|0.8128|
|6|First|
|6|Cnn|||
|8|First||||
|8|Cnn|||


- draw each metric line
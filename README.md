### Run the test
```sh
$ python test.py
bert.embeddings.position_ids
bert.embeddings.word_embeddings.weight
bert.embeddings.position_embeddings.weight
bert.embeddings.token_type_embeddings.weight
bert.embeddings.LayerNorm.weight
bert.embeddings.LayerNorm.bias
bert.encoder.layer.i-th_layer.attention.self.query.weight
bert.encoder.layer.i-th_layer.attention.self.query.bias
bert.encoder.layer.i-th_layer.attention.self.key.weight
bert.encoder.layer.i-th_layer.attention.self.key.bias
bert.encoder.layer.i-th_layer.attention.self.value.weight
bert.encoder.layer.i-th_layer.attention.self.value.bias
bert.encoder.layer.i-th_layer.attention.output.dense.weight
bert.encoder.layer.i-th_layer.attention.output.dense.bias
bert.encoder.layer.i-th_layer.attention.output.LayerNorm.weight
bert.encoder.layer.i-th_layer.attention.output.LayerNorm.bias
bert.encoder.layer.i-th_layer.intermediate.dense.weight
bert.encoder.layer.i-th_layer.intermediate.dense.bias
bert.encoder.layer.i-th_layer.output.dense.weight
bert.encoder.layer.i-th_layer.output.dense.bias
bert.encoder.layer.i-th_layer.output.LayerNorm.weight
bert.encoder.layer.i-th_layer.output.LayerNorm.bias
bert.pooler.dense.weight
bert.pooler.dense.bias
cls.predictions.bias
cls.predictions.transform.dense.weight
cls.predictions.transform.dense.bias
cls.predictions.transform.LayerNorm.weight
cls.predictions.transform.LayerNorm.bias
cls.predictions.decoder.weight
cls.predictions.decoder.bias
cls.seq_relationship.weight
cls.seq_relationship.bias
          [CLS] ['.', 'the', ',', ')', '"']
            she ['she', 'he', 'her', 'they', 'who']
          likes ['likes', 'loves', 'enjoys', 'prefers', 'like']
         riding ['riding', 'ride', 'rode', 'rides', 'on']
              a ['a', 'her', 'the', 'riding', 'with']
         [MASK] ['bike', 'horse', 'bicycle', 'motorcycle', 'pony']
        because ['because', 'since', '.', 'and', 'after']
         [MASK] ['she', 'he', 'it', 'everyone', 'her']
          rides ['rides', 'rode', 'ride', 'riding', 'is']
             in ['in', 'during', 'from', 'back', 'since']
         [MASK] ['her', 'his', 'their', 'early', 'the']
      childhood ['childhood', 'time', 'school', 'home', 'age']
              . ['.', ';', '!', ',', '?']
          [SEP] ['she', '.', ',', 'her', 'in']
```

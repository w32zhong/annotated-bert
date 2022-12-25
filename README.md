## Transformer architecture (including the decoder)
![image](https://user-images.githubusercontent.com/1407530/209479273-71bef891-c714-4296-9a7f-37ae0a43c01c.png)

(https://arxiv.org/abs/1706.03762)

## Multi-head attention implementation
![image](https://user-images.githubusercontent.com/1407530/209479477-eb489abd-ce1c-4f92-a168-890237b85a84.png)

## BERT-base parameters
![image](https://user-images.githubusercontent.com/1407530/209481127-1a5a7cb7-4876-4de6-a1bd-96f436ddc817.png)

(https://stackoverflow.com/questions/64485777)

## Run the test
Run `test.py` for twice, one to unmask using Huggingface's implementation and to save its model checkpoint; another one to load the checkpoint and run our own code to reproduce the predictions:
```sh
$ python test.py
Downloading:  100%|██████▋| 420M/420M [01:12<00:00, 6.91MB/s]
Some weights of BertForPreTraining were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['cls.predictions.decoder.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
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

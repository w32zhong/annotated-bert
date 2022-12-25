import os
import re
import torch
import transformers
from transformers import BertTokenizer, BertForPreTraining
from mybert import MyBERT


use_huggingface = not os.path.exists('./test')
text = "She likes riding a [MASK] because [MASK] rides in [MASK] childhood."

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_input = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    if use_huggingface:
        transformers.logging.set_verbosity_warning()
        model = BertForPreTraining.from_pretrained("bert-base-uncased")
        model.save_pretrained('test')
        model.eval()
        output = model(**encoded_input)
        prediction_logits = output['prediction_logits']
    else:
        state_dict = torch.load('test/pytorch_model.bin')
        for key in state_dict.keys():
            m = re.match('bert.encoder.layer.([0-9]+).*', key)
            if m is not None:
                if m.group(1) == '11':
                    print(key.replace('11', 'i-th_layer'))
            else:
                print(key)
        bert = MyBERT()
        bert.load_hf_state_dict(state_dict)
        bert.eval() # to avoid dropout
        hidden_states, predictions = bert(**encoded_input)
        prediction_logits = predictions

sorted_logits = torch.argsort(prediction_logits.squeeze(0), dim=1, descending=True)
for i, words in enumerate(sorted_logits):
    input_tok = encoded_input['input_ids'][0][i]
    input_tok = tokenizer.convert_ids_to_tokens([input_tok])[0]
    top_words = words[:5]
    top_words = tokenizer.convert_ids_to_tokens(top_words)
    print(f'{input_tok:>15}', top_words)

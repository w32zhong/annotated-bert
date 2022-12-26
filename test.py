import os
import re
import torch
import transformers
from transformers import BertTokenizer, BertForPreTraining
from mybert import MyBERT


def visualize_att(layer, tokens, attentions):
    import matplotlib.pyplot as plt
    unit = 3
    fig, axes = plt.subplots(3, 4, figsize = (unit * 4, unit * 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.matshow(attentions[0][i], cmap=plt.cm.Greys)
        ax.set_title(f'layer#{layer+1} head#{i+1}')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens, rotation=0)
        ax.grid()
    fig.tight_layout()
    plt.savefig(f'att_layer{layer+1}.png')
    #plt.show()


def test(args=None):
    use_huggingface = not os.path.exists('./test')
    text = "She likes riding a [MASK] because [MASK] rides in [MASK] childhood."

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

    def hook_func(layer, data):
        #target_layer = args - 1
        #if target_layer == layer:
        visualize_att(layer, tokens, data)

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
            hidden_states, predictions = bert(**encoded_input, hook=hook_func)
            prediction_logits = predictions

    sorted_logits = torch.argsort(prediction_logits.squeeze(0), dim=1, descending=True)
    for i, words in enumerate(sorted_logits):
        input_tok = encoded_input['input_ids'][0][i]
        input_tok = tokenizer.convert_ids_to_tokens([input_tok])[0]
        top_words = words[:5]
        top_words = tokenizer.convert_ids_to_tokens(top_words)
        print(f'{input_tok:>15}', top_words)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(test)

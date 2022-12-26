import torch
import math

vocab_size = 30522
hidden_size = 768
max_position_embeddings = 512
type_vocab_size = 2
hidden_dropout_prob = 0.1
layer_norm_eps = 1e-12
num_hidden_layers = 12
num_attention_heads = 12
attention_probs_dropout_prob = 0.1
intermediate_size = 3072

class MyBertLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.att_head_sz = hidden_size // num_attention_heads
        assert self.att_head_sz * num_attention_heads == hidden_size
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key   = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.att_dropout = torch.nn.Dropout(attention_probs_dropout_prob)

        self.attout_dense     = torch.nn.Linear(hidden_size, hidden_size)
        self.attout_LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attout_dropout   = torch.nn.Dropout(hidden_dropout_prob)

        # point-wise feed forward (broadcast the same linear transform to all positions)
        self.intermediate        = torch.nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = torch.nn.functional.gelu

        self.out_dense     = torch.nn.Linear(intermediate_size, hidden_size)
        self.out_LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout   = torch.nn.Dropout(hidden_dropout_prob)

    def feed_forward_chunk(self, att_out):
        intermediate_output = self.intermediate(att_out)
        intermediate_output = self.intermediate_act_fn(intermediate_output)

        hidden_states = self.out_dense(intermediate_output)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.out_LayerNorm(hidden_states + att_out) # skip link
        return hidden_states

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (num_attention_heads, self.att_head_sz)
        x = x.view(new_x_shape)
        x = x.permute(0, 2, 1, 3)
        # x.shape == (batch_sz, n_heads, seq_len, head_size)
        return x

    def forward(self, inputs, attention_mask, layer_idx, hook):
        # inputs.shape == (batch_sz, seq_len, hidden_size)
        key_layer = self.transpose_for_scores(self.key(inputs))
        value_layer = self.transpose_for_scores(self.value(inputs))
        query_layer = self.transpose_for_scores(self.query(inputs))

        # matmul has broadcast, while @ or mm do not have broadcast!
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        # for large values of att_head_sz, the dot products grow large in magnitude,
        # making the softmax function to have small gradients.
        # Here, use squared att_head_sz to scale down and counteract this effect.
        attention_scores = attention_scores / math.sqrt(self.att_head_sz)
        # attention_scores.shape == (batch_sz, n_heads, seq_len, seq_len)
        attention_scores = attention_scores + attention_mask # broadcast
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.att_dropout(attention_probs)
        hook(layer_idx, attention_probs)

        context_layer = attention_probs @ value_layer
        # context_layer.shape == (batch_sz, n_heads, seq_len, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer.shape == (batch_sz, seq_len, n_heads, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # context_layer.shape == (batch_sz, seq_len, hidden_size)

        logits_layer = self.attout_dense(context_layer)
        logits_layer = self.attout_dropout(logits_layer)
        logits_layer = self.attout_LayerNorm(logits_layer + inputs) # skip link

        layer_output = self.feed_forward_chunk(logits_layer)
        return layer_output

class MyBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(m, n) is equivalent to nn.Embedding(n, m)
        self.word_embeddings = torch.nn.Linear(vocab_size, hidden_size, bias=False)
        self.token_type_embeddings = torch.nn.Linear(type_vocab_size, hidden_size, bias=False)
        self.position_embeddings = torch.nn.Linear(max_position_embeddings, hidden_size, bias=False)
        self.emb_LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.emb_dropout = torch.nn.Dropout(hidden_dropout_prob)

        self.layer = torch.nn.ModuleList(
            [MyBertLayer() for _ in range(num_hidden_layers)]
        )

        self.pretrain_head_dense = torch.nn.Linear(hidden_size, hidden_size)
        self.pretrain_head_actfn = torch.nn.functional.gelu
        self.pretrain_head_LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # the last dense layer maps to vocabulary uses the tie_weights() in modeling_utils.py
        self.pretrain_head_bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, token_type_ids, attention_mask, hook):
        batch_sz, seq_len = input_ids.shape
        position_ids = torch.arange(max_position_embeddings).expand((1, -1))
        position_ids = position_ids[:, 0:seq_len]

        input_ids_onehot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).to(dtype=torch.float32)
        token_type_ids_onehot = torch.nn.functional.one_hot(token_type_ids, num_classes=type_vocab_size).to(dtype=torch.float32)
        position_ids_onehot = torch.nn.functional.one_hot(position_ids, num_classes=max_position_embeddings).to(dtype=torch.float32)

        # or, inputs_embeds = input_ids_onehot @ self.word_embeddings.weight.T
        inputs_embeds = self.word_embeddings(input_ids_onehot)
        token_type_embeddings = self.token_type_embeddings(token_type_ids_onehot)
        position_embeddings = self.position_embeddings(position_ids_onehot)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.emb_LayerNorm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        # [batch_sz, seq_len] -> [batch_sz, 1, 1, seq_len] filled with 0 or -inf
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        hidden_states = embeddings
        for i, layer_module in enumerate(self.layer):
            assert hidden_states.shape == (batch_sz, seq_len, hidden_size)
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                layer_idx=i,
                hook=hook
            )

        # unmasking predictions
        predictions = self.pretrain_head_dense(hidden_states)
        predictions = self.pretrain_head_actfn(predictions)
        predictions = self.pretrain_head_LayerNorm(predictions)
        predictions = predictions @ self.word_embeddings.weight
        predictions = predictions + self.pretrain_head_bias
        return hidden_states, predictions

    def load_hf_state_dict(self, state_dict):
        # Similar to https://github.com/pytorch/pytorch/blob/789b1437e945336f83c915ab2f2dd283ac472191/torch/nn/modules/module.py#L1919
        self.word_embeddings.weight.copy_(
            state_dict['bert.embeddings.word_embeddings.weight'].T)
        self.position_embeddings.weight.copy_(
            state_dict['bert.embeddings.position_embeddings.weight'].T)
        self.token_type_embeddings.weight.copy_(
            state_dict['bert.embeddings.token_type_embeddings.weight'].T)
        self.emb_LayerNorm.weight.copy_(
            state_dict['bert.embeddings.LayerNorm.weight'])
        self.emb_LayerNorm.bias.copy_(
            state_dict['bert.embeddings.LayerNorm.bias'])

        for i in range(len(self.layer)):
            self.layer[i].query.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.query.weight'])
            self.layer[i].query.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.query.bias'])
            self.layer[i].key.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.key.weight'])
            self.layer[i].key.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.key.bias'])
            self.layer[i].value.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.value.weight'])
            self.layer[i].value.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.self.value.bias'])

            self.layer[i].attout_dense.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.output.dense.weight'])
            self.layer[i].attout_dense.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.output.dense.bias'])
            self.layer[i].attout_LayerNorm.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'])
            self.layer[i].attout_LayerNorm.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias'])

            self.layer[i].intermediate.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.intermediate.dense.weight'])
            self.layer[i].intermediate.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.intermediate.dense.bias'])

            self.layer[i].out_dense.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.output.dense.weight'])
            self.layer[i].out_dense.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.output.dense.bias'])

            self.layer[i].out_LayerNorm.weight.copy_(
                state_dict[f'bert.encoder.layer.{i}.output.LayerNorm.weight'])
            self.layer[i].out_LayerNorm.bias.copy_(
                state_dict[f'bert.encoder.layer.{i}.output.LayerNorm.bias'])

        self.pretrain_head_dense.weight.copy_(
            state_dict['cls.predictions.transform.dense.weight'])
        self.pretrain_head_dense.bias.copy_(
            state_dict['cls.predictions.transform.dense.bias'])
        self.pretrain_head_LayerNorm.weight.copy_(
            state_dict['cls.predictions.transform.LayerNorm.weight'])
        self.pretrain_head_LayerNorm.bias.copy_(
            state_dict['cls.predictions.transform.LayerNorm.bias'])
        self.pretrain_head_bias.copy_(
            state_dict['cls.predictions.decoder.bias'])

import sys
from pathlib import Path

import numpy as np

from encoder import get_encoder
from checkpoint import Model

MODEL = '124M'

model = Model(Path('models') / MODEL)
_model = {
    'wte': model.get('model/wte'),
    'wpe': model.get('model/wpe'),
    'ln_f_g': model.get('model/ln_f/g'),
    'ln_f_b': model.get('model/ln_f/b'),
    'layers': [{
        'attn_w': model.get(f'model/h{i}/attn/c_attn/w'),
        'attn_b': model.get(f'model/h{i}/attn/c_attn/b'),
        'attn_proj_w': model.get(f'model/h{i}/attn/c_proj/w'),
        'attn_proj_b': model.get(f'model/h{i}/attn/c_proj/b'),
        'ln_1_g': model.get(f'model/h{i}/ln_1/g'),
        'ln_1_b': model.get(f'model/h{i}/ln_1/b'),
        'ln_2_g': model.get(f'model/h{i}/ln_2/g'),
        'ln_2_b': model.get(f'model/h{i}/ln_2/b'),
        'mlp_w': model.get(f'model/h{i}/mlp/c_fc/w'),
        'mlp_b': model.get(f'model/h{i}/mlp/c_fc/b'),
        'mlp_proj_w': model.get(f'model/h{i}/mlp/c_proj/w'),
        'mlp_proj_b': model.get(f'model/h{i}/mlp/c_proj/b'),
    } for i in range(model.hparams['n_layer'])]
}


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def attention(q, k, v):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) * np.tri(q.shape[0])) @ v


def generate(inputs):
    x = _model['wte'][inputs] + _model['wpe'][range(len(inputs))]

    for layer in _model['layers']:
        # attention
        y = layer_norm(x, layer['ln_1_g'], layer['ln_1_b'])
        y = y @ layer['attn_w'] + layer['attn_b']
        q, k, v = np.split(y, 3, axis=-1)
        y = attention(q, k, v)
        y = y @ layer['attn_proj_w'] + layer['attn_proj_b']
        x = x + y

        # feed forward
        y = layer_norm(x, layer['ln_2_g'], layer['ln_2_b'])
        y = y @ layer['mlp_w'] + layer['mlp_b']
        y = gelu(y)
        y = y @ layer['mlp_proj_w'] + layer['mlp_proj_b']
        x = x + y

    x = layer_norm(x, _model['ln_f_g'], _model['ln_f_b'])
    x = x @ _model['wte'].T
    return np.argmax(x[-1])


if __name__ == '__main__':
    prompt = sys.argv[1]
    enc = get_encoder(MODEL, 'models')
    inputs = enc.encode(prompt)
    for i in range(10):
        next_id = generate(inputs)
        inputs.append(next_id)
        print(enc.decode(inputs))

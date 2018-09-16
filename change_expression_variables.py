import json

with open('expression_tensors.json', encoding='utf-8') as f:
    data = json.loads(f.read())

# define maps
name_map = {
    'Variable:0': 'expression_conv1/kernel:0',
    'Variable_1:0': 'expression_conv1/bias:0',
    'Variable_2:0': 'expression_conv2/kernel:0',
    'Variable_3:0': 'expression_conv2/bias:0',
    'Variable_4:0': 'expression_dense1/kernel:0',
    'Variable_5:0': 'expression_dense1/bias:0',
    'Variable_6:0': 'expression_dense2/kernel:0',
    'Variable_7:0': 'expression_dense2/bias:0',
    'Variable_8:0': 'expression_logits/kernel:0',
    'Variable_9:0': 'expression_logits/bias:0',
}

tensors = {}
for k, v in name_map.items():
    tensors[v] = data[k]

# define new tensors
print(tensors.keys())

with open('expression_tensors2.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tensors, ensure_ascii=False))

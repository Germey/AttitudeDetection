import json

with open('head_pose_tensors.json', encoding='utf-8') as f:
    data = json.loads(f.read())

# define maps
name_map = {
    'conv2d/kernel:0': 'landmark_conv1/kernel:0',
    'conv2d/bias:0': 'landmark_conv1/bias:0',
    'conv2d_1/kernel:0': 'landmark_conv2/kernel:0',
    'conv2d_1/bias:0': 'landmark_conv2/bias:0',
    'conv2d_2/kernel:0': 'landmark_conv3/kernel:0',
    'conv2d_2/bias:0': 'landmark_conv3/bias:0',
    'conv2d_3/kernel:0': 'landmark_conv4/kernel:0',
    'conv2d_3/bias:0': 'landmark_conv4/bias:0',
    'conv2d_4/kernel:0': 'landmark_conv5/kernel:0',
    'conv2d_4/bias:0': 'landmark_conv5/bias:0',
    'conv2d_5/kernel:0': 'landmark_conv6/kernel:0',
    'conv2d_5/bias:0': 'landmark_conv6/bias:0',
    'conv2d_6/kernel:0': 'landmark_conv7/kernel:0',
    'conv2d_6/bias:0': 'landmark_conv7/bias:0',
    'conv2d_7/kernel:0': 'landmark_conv8/kernel:0',
    'conv2d_7/bias:0': 'landmark_conv8/bias:0',
    'dense/kernel:0': 'landmark_dense/kernel:0',
    'dense/bias:0': 'landmark_dense/bias:0',
    'logits/kernel:0': 'landmark_logits/kernel:0',
    'logits/bias:0': 'landmark_logits/bias:0'
}

tensors = {}
for k, v in name_map.items():
    tensors[v] = data[k]
    
# define new tensors
print(tensors.keys())

with open('landmark_tensors.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tensors, ensure_ascii=False))

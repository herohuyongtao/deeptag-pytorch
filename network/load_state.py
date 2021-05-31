import collections


def load_state(net, source_state, is_allow_partial_load = True):
    # source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        elif target_key in source_state and is_allow_partial_load and len(source_state[target_key].shape) == len(target_state[target_key].shape):         

            new_target_state[target_key] = target_state[target_key]
            partially_copy_param(source_state[target_key], new_target_state[target_key])
            print('[WARNING] Partially loaded pre-trained parameters for {}'.format(target_key))
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

def partially_copy_param(source_param, target_param):
    s = []
    for s1, s2 in zip(source_param.shape, target_param.shape):
        s.append(min(s1,s2))


    if len(source_param.shape) ==1:
        target_param[:s[0]] = source_param[:s[0]]

    if len(source_param.shape) ==2:
        target_param[:s[0], :s[1]] = source_param[:s[0], :s[1]]

    if len(source_param.shape) ==3:
        target_param[:s[0], :s[1], :s[2]] = source_param[:s[0], :s[1], :s[2]]

    if len(source_param.shape) ==4:
        target_param[:s[0], :s[1], :s[2], :s[3]] = source_param[:s[0], :s[1], :s[2], :s[3]]

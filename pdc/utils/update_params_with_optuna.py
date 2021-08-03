import copy


def update_params_with_optuna(params, trial_params):
    ret_params = copy.deepcopy(params)
    for base_key, value in trial_params.items():
        remaining_key = base_key
        sub_params = ret_params
        next_key_end = remaining_key.find("/")
        while next_key_end > 0:
            key = remaining_key[:next_key_end]
            if key not in sub_params:
                sub_params[key] = {}
            sub_params = sub_params[key]
            remaining_key = remaining_key[next_key_end + 1:]
            next_key_end = remaining_key.find("/")
        key = remaining_key
        sub_params[key] = value
    return ret_params

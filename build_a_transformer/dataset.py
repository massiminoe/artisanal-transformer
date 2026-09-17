# Let's learn about addition, subtraction ?!

import torch

import conf


def create_example(max_abs_val: int = conf.MAX_ABS_VAL) -> str:
    nums = torch.randint(low=-max_abs_val, high=max_abs_val, size=(2,))
    num_1 = nums[0].item()
    num_2 = nums[1].item()

    # addition
    if torch.rand(1) > 0.5:
        if num_2 >= 0:
            return f"{num_1}+{num_2}={num_1+num_2}"
        else:  # num_2 will have the - symbol
            return f"{num_1}{num_2}={num_1+num_2}"

    # substraction
    if num_2 >= 0:
        return f"{num_1}-{num_2}={num_1-num_2}"
    else:  # num_2 will have the - symbol
        return f"{num_1}{num_2}={num_1-num_2}"


def generate_dataset(n: int, max_abs_val: int = conf.MAX_ABS_VAL) -> list[str]:
    dataset = []
    for _ in range(n):
        dataset.append(create_example(max_abs_val))
    return dataset

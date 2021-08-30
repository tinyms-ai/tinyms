"""
Model related utils.
"""


def print_network(network, name):
    num_params = 0
    for _, p in network.parameters_and_names():
        num_params += p.size
    print(f"Parameter number of {name}: {num_params/1e6:.4f}M")

from pprint import pprint


def print_config(config):
    pprint({attr: config.__getattribute__(attr) for attr in dir(config) if not attr.startswith('__')})

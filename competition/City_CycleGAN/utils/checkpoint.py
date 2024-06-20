import os
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint


class CheckpointIO:
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def save(self, step):
        for name, module in self.module_dict.items():
            fname = self.fname_template.format(step, name)
            print('Saving checkpoint into %s...' % fname)
            save_checkpoint(module, fname)

    def load(self, step):
        for name, module in self.module_dict.items():
            fname = self.fname_template.format(step, name)
            assert os.path.exists(fname), fname + ' does not exist!'
            print('Loading checkpoint from %s...' % fname)
            param = load_checkpoint(fname)
            load_param_into_net(module, param)

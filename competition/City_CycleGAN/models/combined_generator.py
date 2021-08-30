from tinyms import layers
from tinyms.primitives import OnesLike


class CombinedGenerator(layers.Layer):
    """
    Generator of CycleGAN, return fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.

    Args:
        G_A (layers.Layer): The generator network of domain A to domain B.
        G_B (layers.Layer): The generator network of domain B to domain A.
        use_identity (bool): Use identity loss or not. Default: True.

    Returns:
        Tensors, fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.
    """

    def __init__(self, G_A, G_B, use_identity=True):
        super().__init__()
        self.G_A = G_A
        self.G_B = G_B
        self.ones = OnesLike()
        self.use_identity = use_identity

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A = self.G_B(img_B)
        fake_B = self.G_A(img_A)
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        if self.use_identity:
            identity_A = self.G_B(img_A)
            identity_B = self.G_A(img_B)
        else:
            identity_A = self.ones(img_A)
            identity_B = self.ones(img_B)
        return fake_A, fake_B, rec_A, rec_B, identity_A, identity_B

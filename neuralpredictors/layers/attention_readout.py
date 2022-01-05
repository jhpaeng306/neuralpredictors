import warnings
from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import ModuleDict
from torch.nn import Parameter
from torch.nn import functional as F

from ..constraints import positive
from ..utils import PositionalEncoding2D
from einops import rearrange


class ConfigurationError(Exception):
    pass


class Attention2d(nn.Module):
    """
    A readout using a transformer layer with self attention.

    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.

        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]

    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        use_pos_enc=True,
        learned_pos=False,
        dropout_pos=0.1,
        **kwargs,
    ):

        super().__init__()

        # determines whether the Gaussian is isotropic or not

        # store statistics about the images and neurons
        self.in_shape = in_shape
        self.outdims = outdims
        self.use_pos_enc = use_pos_enc

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize(learned_pos, dropout_pos)

    @property
    def features(self):
        return self._features

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self._features.abs().mean()
        else:
            return self._features.abs().sum()

    def query_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if average:
            return self.neuron_query.abs().mean()
        else:
            return self.neuron_query.abs().sum()

    def initialize(self, learned_pos, dropout_pos):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        c, h, w = self.in_shape
        self._features = Parameter(torch.Tensor(1, c, self.outdims))
        self._features.data.fill_(1 / self.in_shape[0])

        self.neuron_query = Parameter(torch.Tensor(1, c, self.outdims))
        self.neuron_query.data.fill_(1 / self.in_shape[0])
        if self.use_pos_enc:
            self.position_embedding = PositionalEncoding2D(
                d_model=c, width=w, height=h, learned=learned_pos, dropout=dropout_pos
            )

        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(
        self,
    ):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """

    def forward(self, x, output_attn_weights=False, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            warnings.warn("the specified feature map dimension is not the readout's expected input dimension")
        feat = self.features.view(1, c, self.outdims)
        bias = self.bias
        x = x.flatten(2, 3)
        x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        # compare neuron query with each spatial position (dot-product)
        y = torch.einsum("ics,ocn->isn", x_embed, self.neuron_query)  # -> [Images, w*h, Neurons]
        # compute attention weights
        attention_weights = torch.nn.functional.softmax(y, dim=1)  # -> [Images, w*h, Neurons]
        # compute average weighted with attention weights
        y = torch.einsum("ics,isn->icn", x, attention_weights)  # -> [Images, Channels, Neurons]
        y = torch.einsum("icn,ocn->in", y, feat)  # -> [Images, Neurons]

        if self.bias is not None:
            y = y + bias

        if output_attn_weights:
            return y, attention_weights

        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = " "
        r += self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
        if self.bias is not None:
            r += " with bias"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


class MultiHeadAttention2d(Attention2d):
    """
    A readout using a transformer layer with self attention.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        use_pos_enc=True,
        learned_pos=False,
        heads=1,
        scale=False,
        key_embedding=False,
        value_embedding=False,
        temperature=(False, 1.0),  # (learnable-per-neuron, value)
        dropout_pos=0.1,
        layer_norm=False,
        **kwargs,
    ):

        self.heads = heads
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        super().__init__(in_shape, outdims, bias, use_pos_enc, learned_pos, dropout_pos)
        c, w, h = in_shape
        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(c, c * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(c, c, bias=False)
        if scale:
            dim_head = c // self.heads
            self.scale = dim_head ** -0.5  # prevent softmax gradients from vanishing (for large dim_head)
        else:
            self.scale = 1.0
        if temperature[0]:
            self.T = temperature[1]
        else:
            self.T = Parameter(torch.ones(outdims) * temperature[1])
        if layer_norm:
            self.norm = nn.LayerNorm((c, w * h))
        else:
            self.norm = None

    def forward(self, x, output_attn_weights=False, **kwargs):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
        Returns:
            y: neuronal activity
        """
        i, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            warnings.warn("the specified feature map dimension is not the readout's expected input dimension")

        x = x.flatten(2, 3)  # [Images, Channels, w*h]
        if self.use_pos_enc:
            x_embed = self.position_embedding(x)  # -> [Images, Channels, w*h]
        else:
            x_embed = x

        if self.norm is not None:
            x_embed = self.norm(x_embed)

        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(x_embed, "i c s -> (i s) c")).chunk(2, dim=-1)
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(value, "(i s) (h d) -> i h d s", h=self.heads, i=i)
        elif self.key_embedding:
            key = self.to_key(rearrange(x_embed, "i c s -> (i s) c"))
            key = rearrange(key, "(i s) (h d) -> i h d s", h=self.heads, i=i)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
        else:
            key = rearrange(x_embed, "i (h d) s -> i h d s", h=self.heads)
            value = rearrange(x, "i (h d) s -> i h d s", h=self.heads)
        query = rearrange(self.neuron_query, "o (h d) n -> o h d n", h=self.heads)

        # compare neuron query with each spatial position (dot-product)
        dot = torch.einsum("ihds,ohdn->ihsn", key, query)  # -> [Images, Heads, w*h, Neurons]
        dot = dot * self.scale / self.T
        # compute attention weights
        attention_weights = torch.nn.functional.softmax(dot, dim=2)  # -> [Images, Heads, w*h, Neurons]
        # compute average weighted with attention weights
        y = torch.einsum("ihds,ihsn->ihdn", value, attention_weights)  # -> [Images, Heads, Head_Dim, Neurons]
        y = rearrange(y, "i h d n -> i (h d) n")  # -> [Images, Channels, Neurons]

        feat = self.features.view(1, c, self.outdims)
        y = torch.einsum("icn,ocn->in", y, feat)  # -> [Images, Neurons]

        if self.bias is not None:
            y = y + self.bias

        if output_attn_weights:
            return y, attention_weights
        return y

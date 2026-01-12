import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder

class MultimodalSourceEncoding(nn.Module):
    def __init__(self, price_dim, macro_dim, dim):
        super().__init__()

        self.macro_encoder = MacroIndicatorEncoder(
            in_dim=macro_dim,
            dim=dim
        )

        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        
        self.news_encoder = NewsEncoder(
            hidden_dim=dim
        )

    def forward(self, s_o, s_h, s_c, s_m, s_n):
        """
        s_o, s_h, s_c: (T,1) hoặc (batch,T,1)
        s_m: (T,num_macro) hoặc (batch,T,num_macro)
        s_n: (T,news_dim)
        """
        v_i = self.indicator_encoder(s_o, s_h, s_c)
        v_m = self.macro_encoder(s_m)
        v_n = self.news_encoder(s_n)

        return v_m, v_i, v_n


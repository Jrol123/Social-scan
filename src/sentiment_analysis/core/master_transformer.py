from pandas import concat

from .config import MasterTransformerConfig
from ..abstract import Transformer

class MasterTransformer:
    def __init__(self, config: MasterTransformerConfig) -> None:
        self.config = config
        
    def transform(self, *transformers: Transformer):
        res = []
        for transformer in transformers:
            res.append(transformer.transform(self.config))
        return concat(res)
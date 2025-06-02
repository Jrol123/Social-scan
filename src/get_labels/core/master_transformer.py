from pandas import concat

from .config import MasterTransformerConfig
from ..abstract import AbstractTransformer

class MasterTransformer:
    def __init__(self, config: MasterTransformerConfig) -> None:
        self.config = config
        
    def transform(self, *transformers: AbstractTransformer):
        res = []
        for transformer in transformers:
            print(transformer.__class__.__name__)
            res.append(transformer.transform(self.config))
        return concat(res)
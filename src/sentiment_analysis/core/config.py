from pandas import DataFrame
from ...abstract import GlobalConfig



class MasterTransformerConfig(GlobalConfig):
    df: DataFrame
    
    # TODO: Разделять df на семантический и несемантический
    # Возможно, надо будет добавить доп. инфу в конфиг (.txt).

from tgis_parser import TGisParser

# abc = TGisParser("gornoaltaysk", 70000001070071240) # TODO: Тестировать на более сложных конструкциях
abc = TGisParser("crimea", 70000001046404911)
abc.parse()
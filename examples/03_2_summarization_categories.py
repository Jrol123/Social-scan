"""
Пример многоклассовой суммаризации
"""
import os

os.add_dll_directory(r"D:\Hardcode_libs\GTK3-Runtime Win64\bin")

import pandas as pd
from dotenv import dotenv_values

from src.get_summarization import gen_multilabel_summarization, gen_categories

if __name__ == "__main__":

    secrets = dotenv_values()

    data = pd.read_csv("examples/02a_example_filtered_data.csv")
    metadata = {
        "company": "МРИЯ",
        "description": "курорт премиум-класса на южном берегу Крыма",
    }
    classes = gen_categories(data, "mistral", secrets['MISTRAL_API_TOKEN'], metadata)
    # classes = ['Качество номеров', 'Сервис и обслуживание', 'Питание',
    #            'Инфраструктура и развлечения', 'Ценообразование и компенсации',
    #            'Общее впечатление и атмосфера', 'Здоровье и безопасность',
    #            'Коммуникация и информация', 'Локация и транспорт',
    #            'Управление и руководство', 'Проблемы с питанием',
    #            'Проблемы с обслуживанием и персоналом',
    #            'Проблемы с инфраструктурой и номерами',
    #            'Проблемы с организацией отдыха', 'Проблемы с ценовой политикой',
    #            'Проблемы с безопасностью и конфиденциальностью',
    #            'Проблемы с организацией мероприятий',
    #            'Проблемы с коммуникацией и обратной связью',
    #            'Проблемы с организацией отдыха для семей с детьми',
    #            'Проблемы с организацией отдыха для VIP-гостей',
    #            'Организация питания и ресторанов', 'Качество номеров и уборка',
    #            'Персонал и сервис', 'Организация территории и навигация',
    #            'Ценообразование и дополнительные услуги',
    #            'Развлечения и инфраструктура', 'Общее впечатление и рекомендации']
    
    problems = gen_multilabel_summarization(
        data,
        classes,
        metadata,
        secrets["CHUTES_API_TOKEN"],
        model_name="deepseek",
    )
    problems = pd.DataFrame(problems)
    problems.to_csv("examples/03_1_summarized_data_categories.csv")

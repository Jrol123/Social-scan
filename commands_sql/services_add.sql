INSERT INTO Services (Service_name, parsing_config)
VALUES (
        'Yandex',
        json(
            '
        {
            "url": "https://yandex.ru/maps/",
            "input_xpath": ".//input[@class=''input__control _bold'']",
            "confirm_xpath": ".//button[@class=''button _view_search _size_medium'']",
            "card_xpath": ".//li[@class=''search-snippet-view'']",
            "url_pos": [-2, -2] 
        }
    '
        )
    );
INSERT INTO Services (Service_name, parsing_config)
VALUES (
        'Google',
        json(
            '
        {
            "url": "https://www.google.com/maps/",
            "input_xpath": ".//input",
            "confirm_xpath": ".//button[@id=''searchbox-searchbutton'']",
            "card_xpath": ".//a",
            "url_pos": [-3, -1]
        }
    '
    --! TODO: у Google постоянно меняются xpath. Возможно, стоит брать по типу (сейчас нужно брать третий <a> и первый инпут)
    -- TODO: Попробовать поработать с google.com/maps/search и с его аналогом у Yandex
        )
    );
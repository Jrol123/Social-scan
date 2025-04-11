from abc import ABC, abstractmethod


class Parser(ABC):
    @abstractmethod
    def parse(self, q: str | list[str]) -> dict[str, str | int | float | None]:
        """
        Получение информации с сервиса по запросу.
        
        Args:
            q (str | list[str]): Информация, необходимая для поиска объекта в сервисе.

        Returns:
            dict[str,str|int|float|None]: Словарь с данными. Структура:
            ```
            {
                "name" (str): Имя пользователя. Для Telegram и VK хранить id пользователя.
                "additional_id" (str | None): Дополнительный идентификатор для уточнения сообщения (пример: канал в Telegram).
                "date" (int): Дата в формате timestamp.
                "rating" (float | None): Рейтинг (1.0-5.0, если есть, иначе None).
                "text" (str): Текст отзыва.
                "answer" (str | None): Ответ на отзыв (если присутствует).
            }
            ```
                
        Examples:
            **Вывод с карт:**
            ```
            {
                "name": "Иван Иванов",
                "additional_id": None,
                "date": 1678901234,
                "rating": 5.0,
                "text": "Отличный сервис!",
                "answer": "Спасибо за отзыв!",
            }
            ```
            
            **Вывод с Telegram:**
            ```
            {
                "name": "123",
                "additional_id": "123",
                "date": 1678901234,
                "rating": None,
                "text": "Отличный сервис!",
                "answer": None,
            }
            ```
        
        """
        pass

DROP TABLE IF EXISTS Services;
DROP TABLE IF EXISTS Authors;
DROP TABLE IF EXISTS Messages;

-- Таблица сервисов
CREATE TABLE IF NOT EXISTS Services(
    id INTEGER PRIMARY KEY,
    Service_name VARCHAR(30) NOT NULL,
    parsing_config JSON, -- JSON с XPath, URL и другими параметрами
    last_parsed_at INTEGER DEFAULT NONE, -- Timestamp последнего парсинга
    last_timestamp INTEGER  -- Timestamp последнего сообщения  -- Можно будет брать комментарии по времени, игнорируя авторов, которые уже добавлены в бд (для карт)
);

-- Таблица авторов
CREATE TABLE IF NOT EXISTS Authors(
    id INTEGER PRIMARY KEY,
    Service_id INTEGER NOT NULL, -- Привязка к сервису
    Full_name TEXT,
    -- UNIQUE(Service_id, Full_name), -- Уникальный автор в рамках сервиса  -- Не подходит для мессенджеров
    FOREIGN KEY (Service_id) REFERENCES Services(id)
);

-- Таблица комментариев
CREATE TABLE IF NOT EXISTS Messages(
    id INTEGER PRIMARY KEY,
    id_inService INTEGER DEFAULT NONE,  -- id в сервисе, если сервис использует свою идентификацию сообщений
    Service INTEGER NOT NULL,
    Author INTEGER NOT NULL,
    Timestamp INTEGER,
    Message TEXT,
    Updated_at INTEGER,  -- время последнего обновления комментария
    -- UNIQUE(Service, Author), -- Один комментарий от автора на сервис  -- Не подходит для мессенджеров
    FOREIGN KEY(Author) REFERENCES Authors(id),
    FOREIGN KEY(Service) REFERENCES Services(id)
);

UPDATE Services 
SET parsing_config = json('{
  "base_url": "https://example.com/comments",
  "comments_xpath": "//div[@class='message']",
  "pagination": {"selector": "a.next", "type": "xpath"}
}')
WHERE id = 1;
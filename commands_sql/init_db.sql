DROP TABLE IF EXISTS Services;
DROP TABLE IF EXISTS Authors;
DROP TABLE IF EXISTS Messages;

-- Таблица сервисов
CREATE TABLE IF NOT EXISTS Services(
    id INTEGER PRIMARY KEY,
    Service_name VARCHAR(30) NOT NULL,
    parsing_config JSON, -- JSON с XPath, URL и другими параметрами
    last_parsed_at INTEGER DEFAULT NONE, -- Timestamp последнего парсинга
    last_timestamp INTEGER DEFAULT NONE  -- Timestamp последнего сообщения  -- Можно будет брать комментарии по времени, игнорируя авторов, которые уже добавлены в бд (для карт)
);

-- Таблица авторов
CREATE TABLE IF NOT EXISTS Authors(
    id INTEGER PRIMARY KEY,
    Service_id INTEGER NOT NULL, -- Привязка к сервису
    Full_name TEXT NOT NULL,
    -- UNIQUE(Service_id, Full_name), -- Уникальный автор в рамках сервиса  -- Не подходит для мессенджеров
    FOREIGN KEY (Service_id) REFERENCES Services(id)
);

-- Таблица комментариев
CREATE TABLE IF NOT EXISTS Messages(
    id INTEGER PRIMARY KEY,
    id_inService INTEGER NOT NULL DEFAULT NONE,  -- id в сервисе, если сервис использует свою идентификацию сообщений
    Service_id INTEGER NOT NULL,
    Author_id INTEGER NOT NULL,
    Timestamp INTEGER NOT NULL,
    Message TEXT,
    -- Updated_at INTEGER,  -- время последнего обновления комментария  -- Не будем переопределять комментарии
    -- UNIQUE(Service, Author), -- Один комментарий от автора на сервис  -- Не подходит для мессенджеров
    FOREIGN KEY(Author_id) REFERENCES Authors(id),
    FOREIGN KEY(Service_id) REFERENCES Services(id)
);

CREATE TABLE IF NOT EXISTS
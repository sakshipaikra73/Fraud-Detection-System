-- schema.sql
CREATE TABLE IF NOT EXISTS scanned_urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    confidence REAL NOT NULL,
    risk_level TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    risk_factors TEXT
);

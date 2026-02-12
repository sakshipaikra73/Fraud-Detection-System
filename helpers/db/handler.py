import sqlite3
import os
import contextlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'database', 'database.db')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'database', 'schema.sql')

class DBHandler:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    @contextlib.contextmanager
    def get_cursor(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn.cursor()
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_db(self):
        """Initialize the database with schema."""
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path))
            
        with open(SCHEMA_PATH, 'r') as f:
            schema_script = f.read()
            
        with self.get_cursor() as cursor:
            cursor.executescript(schema_script)
            print("Database initialized successfully.")

    def insert_scan_result(self, url, is_fraud, confidence, risk_level, ip_address=None, risk_factors=None):
        """Insert a new scan record."""
        # Convert list of factors to string if needed, or assume it's passed as string/JSON
        if isinstance(risk_factors, list):
            risk_factors = ", ".join(risk_factors)
            
        query = """
        INSERT INTO scanned_urls (url, is_fraud, confidence, risk_level, ip_address, risk_factors, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (url, is_fraud, confidence, risk_level, ip_address, risk_factors, datetime.now()))

    def fetch_history(self, limit=10, offset=0):
        """Fetch scan history with pagination."""
        query = "SELECT * FROM scanned_urls ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        with self.get_cursor() as cursor:
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_total_count(self):
        """Get total number of scans."""
        query = "SELECT COUNT(*) FROM scanned_urls"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()[0]

if __name__ == "__main__":
    # Initialize DB when run directly
    handler = DBHandler()
    handler.init_db()

# Refactoring the functional code to an OOP structure
import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

RUN_TIMEZONE_CHECK = os.getenv("RUN_TIMEZONE_CHECK", "1") == "1"


class DatabaseManager:
    def __init__(
        self,
        host=None,
        database=None,
        user=None,
        password=None,
        tz_info="Europe/Berlin",
    ):
        """
        Initializes the DatabaseManager with connection details and timezone information.
        """
        self._host = host or os.getenv("POSTGRES_HOST", "postgres")
        self._database = database or os.getenv("POSTGRES_DB", "course_assistant")
        self._user = user or os.getenv("POSTGRES_USER", "your_username")
        self._password = password or os.getenv("POSTGRES_PASSWORD", "your_password")
        self._tz = ZoneInfo(tz_info or os.getenv("TZ", "Europe/Berlin"))
        self._conn = self._get_db_connection()

    def _get_db_connection(self):
        """
        Establishes a connection to the PostgreSQL database.
        """
        return psycopg2.connect(
            host=self._host,
            database=self._database,
            user=self._user,
            password=self._password,
        )

    def init_db(self):
        """
        Initializes the database by creating necessary tables.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")

                cur.execute(
                    """
                    CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                    """
                )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e
        finally:
            self._conn.close()

    def save_conversation(self, conversation_id, question, answer, timestamp=None):
        """
        Saves a conversation to the database.
        """
        if timestamp is None:
            timestamp = datetime.now(self._tz)
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        conversation_id,
                        question,
                        answer["answer"],
                        answer["model_used"],
                        answer["response_time"],
                        answer["relevance"],
                        answer["relevance_explanation"],
                        answer["prompt_tokens"],
                        answer["completion_tokens"],
                        answer["total_tokens"],
                        answer["eval_prompt_tokens"],
                        answer["eval_completion_tokens"],
                        answer["eval_total_tokens"],
                        answer["openai_cost"],
                        timestamp,
                    ),
                )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e
        finally:
            self._conn.close()

    def save_feedback(self, conversation_id, feedback, timestamp=None):
        """
        Saves feedback for a conversation to the database.
        """
        if timestamp is None:
            timestamp = datetime.now(self._tz)
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))",
                    (conversation_id, feedback, timestamp),
                )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e
        finally:
            self._conn.close()

    def get_recent_conversations(self, limit=10, relevance=None):
        """
        Retrieves the most recent conversations from the database.
        """
        try:
            with self._conn.cursor(cursor_factory=DictCursor) as cur:
                query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
                """
                if relevance:
                    query += f" WHERE c.relevance = '{relevance}'"
                    query += " ORDER BY c.timestamp DESC LIMIT %s"

                cur.execute(query, (limit,))
                return cur.fetchall()
        except Exception as e:
            raise e
        finally:
            self._conn.close()

    def get_feedback_stats(self):
        """
        Retrieves statistics about feedback, such as count and average feedback length.
        """
        try:
            with self._conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT 
                        SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                        SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                    FROM feedback
                    """
                )
            return cur.fetchone()
        except Exception as e:
            raise e
        finally:
            self._conn.close()

    def check_timezone(self):
        """
        Checks the current time in the configured timezone and compares it with UTC.
        """
        try:
            with self._conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SHOW timezone;")
            db_timezone = cur.fetchone()
            if db_timezone:
                db_timezone = db_timezone[0]
                print(f"Database timezone: {db_timezone}")
            else:
                print("No timezone found")

            db_time_utc = cur.fetchone()
            if db_time_utc:
                db_time_utc = db_time_utc[0]
                print(f"Database current time (UTC): {db_time_utc}")
            else:
                print("No current time found")

            db_time_local = db_time_utc.astimezone(tz)  # type: ignore
            print(f"Database current time ({self._tz}): {db_time_local}")

            py_time = datetime.now(self._tz)
            print(f"Python current time: {py_time}")

            # Use py_time instead of tz for insertion
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, model_used, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING timestamp;
            """,
                (
                    "test",
                    "test question",
                    "test answer",
                    "test model",
                    0.0,
                    0.0,
                    "test explanation",
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    py_time,
                ),
            )

            inserted_time = cur.fetchone()
            if inserted_time:
                inserted_time = inserted_time[0]
                print(f"Inserted time (UTC): {inserted_time}")
                print(
                    f"Inserted time ({self._tz}): {inserted_time.astimezone(self._tz)}"
                )
            else:
                print("No inserted time found")

            cur.execute("SELECT timestamp FROM conversations WHERE id = 'test';")
            selected_time = cur.fetchone()
            if selected_time:
                selected_time = selected_time[0]
                print(f"Selected time (UTC): {selected_time}")
                print(
                    f"Selected time ({self._tz}): {selected_time.astimezone(self._tz)}"
                )
            else:
                print("No selected time found")

            # Clean up the test entry
            cur.execute("DELETE FROM conversations WHERE id = 'test';")
            self._conn.commit()
        except Exception as e:
            raise e
        finally:
            self._conn.close()

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self._conn:
            self._conn.close()

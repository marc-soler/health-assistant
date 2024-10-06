import os
import uuid
import random
import logging
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

RUN_TIMEZONE_CHECK = os.getenv("RUN_TIMEZONE_CHECK", "1") == "1"


class DatabaseManager:
    def __init__(
        self,
        logger=None,
        host=None,
        database=None,
        user=None,
        password=None,
        tz_info="Europe/Berlin",
    ):
        """
        Initializes the DatabaseManager with connection details and timezone information.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._host = host or os.getenv("POSTGRES_HOST", "postgres")
        self._database = database or os.getenv("POSTGRES_DB", "health_assistant")
        self._user = user or os.getenv("POSTGRES_USER", "your_username")
        self._password = password or os.getenv("POSTGRES_PASSWORD", "your_password")
        self._tz = ZoneInfo(tz_info or os.getenv("TZ", "Europe/Berlin"))
        self._conn = self._get_db_connection()

        self.logger.info(
            f"DatabaseManager initialized with DB: {self._database}, Host: {self._host}"
        )

    def _get_db_connection(self):
        """
        Establishes a connection to the PostgreSQL database.
        """
        try:
            conn = psycopg2.connect(
                host=self._host,
                database=self._database,
                user=self._user,
                password=self._password,
            )
            self.logger.info("Successfully connected to the PostgreSQL database.")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to the PostgreSQL database: {e}")
            raise

    def is_db_initialized(self):
        """
        Checks if the required tables (conversations, feedback) are present in the database.
        Returns True if they exist, False otherwise.
        """
        try:
            with self._conn.cursor() as cur:
                # Check for the existence of the 'conversations' table
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'conversations'
                    );
                """
                )
                conversations_table_exists = cur.fetchone()
                conversations_table_exists = (
                    conversations_table_exists[0]
                    if conversations_table_exists
                    else False
                )

                # Check for the existence of the 'feedback' table
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'feedback'
                    );
                """
                )
                feedback_table_exists = cur.fetchone()
                feedback_table_exists = (
                    feedback_table_exists[0] if feedback_table_exists else False
                )

                # Return True only if both tables exist
                return conversations_table_exists and feedback_table_exists
        except Exception as e:
            self.logger.error(f"Failed to check if database is initialized: {e}")
            raise

    def generate_mock_data(self):
        """
        Generates 100 rows of mock data for the conversations and feedback tables.
        """
        try:
            with self._conn.cursor() as cur:
                self.logger.info("Generating mock data...")

                # Generate 100 mock conversations
                for i in range(100):
                    conversation_id = str(uuid.uuid4())
                    question = f"Mock question {i + 1}"
                    answer = f"Mock answer {i + 1}"
                    model_used = "gpt-40-mini"
                    response_time = round(
                        random.uniform(1.0, 3.0), 2
                    )  # random response time between 1 and 3 seconds
                    relevance = random.choice(
                        ["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
                    )
                    relevance_explanation = f"Mock relevance explanation {i + 1}"
                    prompt_tokens = random.randint(10, 50)
                    completion_tokens = random.randint(10, 50)
                    total_tokens = prompt_tokens + completion_tokens
                    eval_prompt_tokens = random.randint(5, 25)
                    eval_completion_tokens = random.randint(5, 25)
                    eval_total_tokens = eval_prompt_tokens + eval_completion_tokens
                    openai_cost = round(random.uniform(0.01, 0.10), 4)
                    timestamp = datetime.now(self._tz) - timedelta(
                        hours=random.randint(1, 100)
                    )

                    cur.execute(
                        """
                        INSERT INTO conversations (
                            id, question, answer, model_used, response_time, relevance, relevance_explanation, 
                            prompt_tokens, completion_tokens, total_tokens, eval_prompt_tokens, eval_completion_tokens, 
                            eval_total_tokens, openai_cost, timestamp
                        ) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            conversation_id,
                            question,
                            answer,
                            model_used,
                            response_time,
                            relevance,
                            relevance_explanation,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            eval_prompt_tokens,
                            eval_completion_tokens,
                            eval_total_tokens,
                            openai_cost,
                            timestamp,
                        ),
                    )

                    # Generate corresponding feedback for the conversation
                    feedback_value = random.randint(
                        0, 1
                    )  # Random feedback value between 1 and 0
                    feedback_timestamp = timestamp + timedelta(
                        seconds=random.randint(5, 150)
                    )  # Some time after the conversation

                    cur.execute(
                        """
                        INSERT INTO feedback (conversation_id, feedback, timestamp) 
                        VALUES (%s, %s, %s)
                        """,
                        (conversation_id, feedback_value, feedback_timestamp),
                    )

                self._conn.commit()
                self.logger.info("Mock data generated and committed successfully.")
        except Exception as e:
            self._conn.rollback()
            self.logger.error(f"Failed to generate mock data: {e}")
            raise

    def init_db(self):
        """
        Initializes the database by creating necessary tables.
        """
        try:
            with self._conn.cursor() as cur:
                # Drop tables if they exist (safeguard, though they shouldn't exist)
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")
                self.logger.info("Dropped existing tables if they existed.")

                # Create the conversations table
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
                # Create the feedback table
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
            # Commit table creation
            self._conn.commit()
            self.logger.info("Database initialized successfully.")

            # Generate mock data after initializing the tables
            self.generate_mock_data()

        except Exception as e:
            self._conn.rollback()
            self.logger.error(f"Failed to initialize the database: {e}")
            raise

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
            self.logger.info(f"Saved conversation with ID: {conversation_id}.")
        except Exception as e:
            self._conn.rollback()
            self.logger.error(
                f"Failed to save conversation with ID: {conversation_id}. Error: {e}"
            )
            raise

    def save_feedback(self, conversation_id, feedback, timestamp=None):
        """
        Saves feedback for a conversation to the database.
        """
        if timestamp is None:
            timestamp = datetime.now(self._tz)

        try:
            with self._conn.cursor() as cur:
                # Ensure that the conversation ID exists
                cur.execute(
                    "SELECT id FROM conversations WHERE id = %s", (conversation_id,)
                )
                result = cur.fetchone()
                if not result:
                    self.logger.error(
                        f"Conversation ID {conversation_id} does not exist. Cannot save feedback."
                    )
                    raise ValueError(
                        f"Conversation ID {conversation_id} not found in conversations."
                    )

                # Insert feedback if conversation exists
                cur.execute(
                    "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))",
                    (conversation_id, feedback, timestamp),
                )
            self._conn.commit()
            self.logger.info(f"Saved feedback for conversation ID: {conversation_id}.")
        except Exception as e:
            self._conn.rollback()
            self.logger.error(
                f"Failed to save feedback for conversation ID: {conversation_id}. Error: {e}"
            )
            raise

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
                results = cur.fetchall()
                self.logger.info(f"Retrieved {len(results)} recent conversations.")
                return results
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent conversations. Error: {e}")
            raise

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
            stats = cur.fetchone()
            self.logger.info("Retrieved feedback statistics.")
            return stats
        except Exception as e:
            self.logger.error(f"Failed to retrieve feedback statistics. Error: {e}")
            raise

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
                self.logger.info(f"Database timezone: {db_timezone}")
            else:
                self.logger.warning("No timezone found")

            db_time_utc = cur.fetchone()
            if db_time_utc:
                db_time_utc = db_time_utc[0]
                self.logger.info(f"Database current time (UTC): {db_time_utc}")
            else:
                self.logger.warning("No current time found")

            db_time_local = db_time_utc.astimezone(tz)  # type: ignore
            self.logger.info(f"Database current time ({self._tz}): {db_time_local}")

            py_time = datetime.now(self._tz)
            self.logger.info(f"Python current time: {py_time}")

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
                self.logger.info(f"Inserted time (UTC): {inserted_time}")
                self.logger.info(
                    f"Inserted time ({self._tz}): {inserted_time.astimezone(self._tz)}"
                )
            else:
                self.logger.warning("No inserted time found")

            cur.execute("SELECT timestamp FROM conversations WHERE id = 'test';")
            selected_time = cur.fetchone()
            if selected_time:
                selected_time = selected_time[0]
                self.logger.info(f"Selected time (UTC): {selected_time}")
                self.logger.info(
                    f"Selected time ({self._tz}): {selected_time.astimezone(self._tz)}"
                )
            else:
                self.logger.warning("No selected time found")

            # Clean up the test entry
            cur.execute("DELETE FROM conversations WHERE id = 'test';")
            self._conn.commit()
            self.logger.info("Test entry cleaned up successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while checking the timezone: {e}")
            raise

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self._conn:
            self._conn.close()
            self.logger.info("Database connection closed.")

#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until psql "host=$POSTGRES_HOST user=$POSTGRES_USER password=$POSTGRES_PASSWORD dbname=$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

# Initialize the database
python -c 'from health_assistant.database import DatabaseManager; db_manager = DatabaseManager(); db_manager.init_db(); db_manager.close_connection()'

# Run the application
exec "$@"
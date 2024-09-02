from logging_setup import LoggerSetup
from orchestrator import Orchestrator

if __name__ == "__main__":
    LoggerSetup().setup_logging()

    orchestrator = Orchestrator()
    orchestrator.run()

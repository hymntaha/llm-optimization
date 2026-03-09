#!/usr/bin/env python
import sys
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root (two levels up from this file)
env_path = Path(__file__).parent.parent.parent.parent.parent / '.env'
if not env_path.exists():
    # Fallback: try current working directory
    env_path = Path('.env')
load_dotenv(dotenv_path=env_path, override=True)

from datetime import datetime
from financial_researcher.crew import FinancialResearcher

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
  inputs = {
    'company': 'Tesla',
  }
  try:
    result = FinancialResearcher().crew().kickoff(inputs=inputs)
    print(result.raw)
  except Exception as e:
    raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
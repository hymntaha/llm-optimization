#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from financial_researcher.crew import FinancialResearcher

warnings.filterwarning

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
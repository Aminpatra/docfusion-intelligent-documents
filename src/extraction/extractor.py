import re


def extract_total(text):

  pattern = r"\d+\.\d{2}"

  matches = re.findall(pattern, text)

  if matches:
    return matches[-1]

  return None


def extract_date(text):

  pattern = r"\d{4}-\d{2}-\d{2}"

  matches = re.findall(pattern, text)

  if matches:
    return matches[0]

  return None
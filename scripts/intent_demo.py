import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Readable multi-scenario demo for Milestone 7 intent extraction.

Type: Rule-based (demo script only, no model of its own)
"""

from ai_nlp.intent_extraction.intent_parser import extract_intent

DEMO_QUERIES = [
    "running shoes under 4000 rupees, lightweight and environmentally responsible",
    "jeans between 2000 and 5000",
    "recycled cotton bag below Rs 3000",
    "sustainable jacket",
    "electronics under 1000",
    "a nice dress",
]


def main():
    for query in DEMO_QUERIES:
        result = extract_intent(query)
        print("=" * 70)
        print(f"Query: {query}")
        for key, value in result.items():
            if key == "raw_query":
                continue
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

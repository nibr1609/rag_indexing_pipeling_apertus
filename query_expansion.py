#!/usr/bin/env python3
"""
Query Expansion Module

Expands user queries to be more specific and contextually relevant for retrieval.
Uses OpenAI-compatible API to reformulate questions for better search results.

For example:
- "does the family allowance count for adopted children"
  -> "Do employees at ETH Zürich receive family allowance benefits for adopted children?
      What are the employment policies regarding family benefits and adoption at ETH Zürich?"
"""

import os
import openai
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def expand_query(
    query: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    context: str = "ETH Zürich administration and employment",
    verbose: bool = False
) -> str:
    """
    Expand a user query to be more specific and contextually relevant.

    Args:
        query (str): Original user query
        api_key (str, optional): OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url (str, optional): API base URL (defaults to OPENAI_BASE_URL env var)
        model (str, optional): Model to use (defaults to QUERY_EXPANSION_MODEL env var)
        context (str): Context for query expansion (default: ETH Zürich)
        verbose (bool): Whether to print debug information

    Returns:
        str: Expanded query optimized for retrieval
    """
    # Get credentials from environment if not provided
    if api_key is None:
        api_key = os.getenv("QUERY_EXPANSION_API_KEY")
    if base_url is None:
        base_url = os.getenv("QUERY_EXPANSION_BASE_URL", "https://api.swissai.cscs.ch/v1")
    if model is None:
        model = os.getenv("QUERY_EXPANSION_MODEL", "gpt-4")

    if not api_key:
        raise ValueError(
            "API key is required for query expansion.\n"
            "Please provide api_key parameter or set QUERY_EXPANSION_API_KEY environment variable."
        )

    if verbose:
        print("=" * 70)
        print("Query Expansion")
        print("=" * 70)
        print(f"Original query: {query}")
        print(f"Model: {model}")
        print(f"Context: {context}")
        print("=" * 70)

    # Create OpenAI client
    client = openai.Client(api_key=api_key, base_url=base_url)

    # Create system prompt for query expansion
    system_prompt = f"""You are a query-expansion assistant for a document retrieval system.

Task: Reformulate the user's question into a SHORT expanded query for document retrieval. Use "{context}" as the domain.

CRITICAL RULES:
- Maximum length: 2-3 SHORT sentences (30-50 words total MAX)
- Mix question + declarative forms
- Add relevant context and synonyms only
- Return ONLY the query - no explanations
- Do NOT write paragraphs or detailed explanations
- Answer in the same language as the query was asked!!!
- Answer in the same language as the query was asked!!!
- Antworte in der selben Sprache in der du gefragt wurdest!!!
- Antworte in der selben Sprache in der du gefragt wurdest!!!

Examples:

User: "does the family allowance count for adopted children"
Assistant: "Family allowance benefits for adopted children at ETH Zürich. Do adopted children qualify for employee family allowances?"

User: "how do I apply for sabbatical leave"
Assistant: "Sabbatical leave application process at ETH Zürich. How to apply for sabbatical as ETH employee?"

User: "parking options on campus"
Assistant: "ETH Zürich campus parking facilities and permits. Employee parking options and availability."

User: "Kann ich remote aus einem anderen Land arbeiten"
Assistant: "Wenn ich aus einem anderen Land remote für ETH Zürich arbeiten möchte, welche Voraussetzungen und Verfahren gelten? Wie kann ich als Mitarbeiter remote arbeiten, wenn ich nicht in der Schweiz bin?"
"""

    user_prompt = query

    # Call API with streaming
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.3,  # Lower temperature for more focused expansion
        )

        # Collect streamed response
        expanded_query = ""
        if verbose:
            print("\nExpanded query: ", end="", flush=True)

        for chunk in res:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                expanded_query += content
                if verbose:
                    print(content, end="", flush=True)

        if verbose:
            print()
            print("=" * 70)

        # Clean up the expanded query
        expanded_query = expanded_query.strip()

        # Fallback to original query if expansion failed
        if not expanded_query:
            if verbose:
                print("Warning: Query expansion returned empty result, using original query")
            return query

        return expanded_query

    except Exception as e:
        if verbose:
            print(f"Error during query expansion: {e}")
            print("Falling back to original query")
        # Return original query if expansion fails
        return query


def main():
    """Example usage."""
    # Example queries for ETH Zürich admin chat
    test_queries = [
        "does the family allowance count for adopted children",
        "how do I apply for sabbatical leave",
        "what are the parking options on campus",
        "Kann ich remote aus einem anderen Land arbeiten",
    ]

    print("=" * 70)
    print("Query Expansion Examples")
    print("=" * 70)

    for query in test_queries:
        print(f"\nOriginal: {query}")
        expanded = expand_query(query, verbose=False)
        print(f"Expanded: {expanded}")
        print("-" * 70)


if __name__ == "__main__":
    main()

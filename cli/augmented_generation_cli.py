import argparse
from lib.augmentation import augment, summarize, cite

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")


    summarize_parser = subparsers.add_parser("summarize", help="Perform summary")
    summarize_parser.add_argument("query", type=str, help="Give search query to provide summary for.")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of search results to summarize")

    citation_parser = subparsers.add_parser("citations", help="gives citation")
    citation_parser.add_argument("query", type=str, help="give search query ")
    citation_parser.add_argument("--limit", type=int, default=5, help="limit to search results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            augment(query, limit=5)

        case "summarize":

            summarize(args.query, args.limit)

        case "citations":
            cite(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
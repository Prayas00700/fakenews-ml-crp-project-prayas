"""
src/inference/predict.py
─────────────────────────
CLI for running predictions against the deployed endpoint.

Usage:
    python src/inference/predict.py --text "Your news article here"
    python src/inference/predict.py --url  "https://example.com/news-article"
    python src/inference/predict.py --file articles.txt   # one article per line
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.inference import predict, batch_predict


def fetch_text_from_url(url: str) -> str:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text or ""
    except Exception as exc:
        print(f"ERROR: Could not extract text from URL: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake News Detection — inference CLI")
    parser.add_argument("--text",      type=str,   help="News article text to classify")
    parser.add_argument("--url",       type=str,   help="URL of a news article to fetch and classify")
    parser.add_argument("--file",      type=str,   help="Path to a file with one article per line")
    parser.add_argument("--endpoint",  type=str,   default=None, help="SageMaker endpoint name")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Classification threshold (default 0.5)")
    parser.add_argument("--region",    type=str,   default=None, help="AWS region")
    args = parser.parse_args()

    kwargs = dict(endpoint_name=args.endpoint, threshold=args.threshold, region=args.region)

    if args.text:
        result = predict(args.text, **kwargs)
        print(json.dumps(result, indent=2))

    elif args.url:
        print(f"Fetching article from: {args.url}")
        text = fetch_text_from_url(args.url)
        if not text:
            print("ERROR: No text extracted from URL.", file=sys.stderr)
            sys.exit(1)
        result = predict(text, **kwargs)
        print(json.dumps(result, indent=2))

    elif args.file:
        lines = Path(args.file).read_text().splitlines()
        texts = [l.strip() for l in lines if l.strip()]
        results = batch_predict(texts, **kwargs)
        for i, (text, res) in enumerate(zip(texts, results), 1):
            print(f"\n[{i}] {text[:120]}…")
            print(f"     → {res['label'].upper()}  (fake_prob={res['fake_prob']:.3f})")

    else:
        # Interactive mode
        print("Fake News Detector — interactive mode (Ctrl+C to quit)\n")
        while True:
            try:
                text = input("Paste article text: ").strip()
                if not text:
                    continue
                result = predict(text, **kwargs)
                print(f"  Prediction: {result['label'].upper()}")
                print(f"  P(fake)   : {result['fake_prob']:.3f}")
                print(f"  P(real)   : {result['real_prob']:.3f}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break


if __name__ == "__main__":
    main()

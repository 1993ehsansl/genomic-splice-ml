"""Download from UCI with SHA256 logging; run: python scripts/download_data.py."""
import argparse
from genosplice.data import download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/raw/splice.data')
    args = parser.parse_args()
    url, sha = download(args.output)
    print(f'URL: {url}\nSHA256: {sha}\nSaved: {args.output}')

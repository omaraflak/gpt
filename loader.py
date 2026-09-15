import os
import epubs
import tarfile
import urllib.request


def download_french_literature(max_books: int = -1) -> str:
    url = "https://huggingface.co/datasets/laion/Project-Gutenberg/resolve/main/french_epub.tar.gz"
    txt_path = "data/french_literature.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    tar_path = "data/french_epub.tar.gz"
    if not os.path.exists(tar_path):
        os.makedirs("data", exist_ok=True)
        print(f"Downloading {url} to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)

    books_count = 0
    with open(txt_path, "w", encoding="utf-8") as out_f:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".epub")):
                    continue
                try:
                    f_obj = tar.extractfile(member)
                    if f_obj is None:
                        continue
                    book_text = epubs.extract_text(f_obj.read())
                    if book_text:
                        if books_count > 0:
                            out_f.write("\n\n")
                        out_f.write(book_text)
                        books_count += 1
                        if books_count % 250 == 0:
                            print(f"Extracted {books_count} books...", flush=True)
                        if max_books > 0 and books_count == max_books:
                            break
                except Exception:
                    continue

    print(f"Finished extracting {books_count} books to {txt_path}.")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

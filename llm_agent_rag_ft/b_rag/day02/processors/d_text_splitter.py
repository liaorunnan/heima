import re


def split_text(text, max_length=512):
    chunks = []

    for para in re.split(r'\n+', text.strip()):
        if not para: continue
        if len(para) <= max_length:
            chunks.append(para)
            continue

        current = [para]
        for delim in ['[。！？……]', '[；]', '[,]']:
            new = []
            for chunk in current:
                if len(chunk) <= max_length:
                    new.append(chunk)
                else:
                    parts = re.split(f'({delim})', chunk)
                    parts = [''.join(parts[i:i + 2]) for i in range(0, len(parts) - 1, 2)]
                    new.extend(parts)
            current = new

        for chunk in current:
            if len(chunk) <= max_length:
                chunks.append(chunk)
            else:
                chunks.extend([chunk[i:i + max_length] for i in range(0, len(chunk), max_length)])

    return [{
        'child': chunk.strip(),
        'parent': (chunks[i - 1] if i > 0 else '') + chunk + (chunks[i + 1] if i < len(chunks) - 1 else '')
    } for i, chunk in enumerate(chunks) if chunk.strip()]


if __name__ == "__main__":
    text = "1" * 510 + "，eeeeeeeee" + "2" * 512
    result = split_text(text, 512)
    print(result)
    for i, item in enumerate(result):
        print(f"块{i + 1}: 子({len(item['child'])}): {item['child']}")
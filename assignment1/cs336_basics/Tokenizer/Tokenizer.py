import regex as re
import os
from typing import Dict, List, Tuple, Iterable, Iterator, BinaryIO, Optional
from collections import Counter, defaultdict
from multiprocessing import Process, Queue, Manager
from tqdm import trange, tqdm
import pickle
from queue import Empty

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(special_tokens: List[bytes]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token

    return vocab


def word2bytes(word: str) -> List[bytes]:
    """
    convert a word to bytes
    """
    bytes_ids = [bytes([b]) for b in word.encode('utf-8')]

    return bytes_ids


def split_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]

    # 逆序排序， 例如<endoftext>在<end>前面，否则<endoftext>可能会被匹配成<end>
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = '|'.join(re.escape(t) for t in special_tokens)
    special_chunks = re.split(f'({pattern})', text)

    return special_chunks


def pre_tokenize_string(text: str, special_tokens: List[str], include_special: bool = False) -> Counter:
    """
    include_special表示是否将special_tokens中的token也加入计数器
    """
    word_counter = Counter()
    special_chunks = split_by_special_tokens(text, special_tokens)

    for chunk in special_chunks:
        if chunk in special_tokens:
            if include_special:
                # 转换为tuple是为了键可哈希
                token = tuple(word2bytes(chunk))
                word_counter[token] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                token = tuple(word2bytes(chunk))
                word_counter[token] += 1

    return word_counter


def pre_tokenize_string_worker(
        input_path: str | os.PathLike,
        special_tokens: List[str],
        queue: Queue,
        start: int,
        end: int,
        include_special: bool = False
):
    with open(input_path, mode='rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')

    word_counter = pre_tokenize_string(chunk, special_tokens, include_special)

    queue.put(word_counter)


def pair_counts(word_counter: Dict[Tuple[bytes], int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    合并相邻的bytes => pair<bytes, bytes>
    """
    cnt: Dict[Tuple[bytes, bytes], int] = {}
    for token, freq in word_counter.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            cnt[pair] = cnt.get(pair, 0) + freq

    return cnt


def get_max_freq_pair(cnt: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    max_freq = max(cnt.values())
    condidates = [pair for pair, freq in cnt.items() if freq == max_freq]
    res = max(condidates)

    return res


def add_pair_to_vocab(
        vocab: Dict[int, bytes],
        pair: Tuple[bytes, bytes],
        vocab_inv: Dict[bytes, int]
) -> int:
    """
    add a new pair to vocab
    """
    index = len(vocab)
    s = vocab[vocab_inv[pair[0]]] + vocab[vocab_inv[pair[1]]]
    vocab[index] = s
    vocab_inv[s] = index

    return index


def merge_pair(
        word_counter: Dict[Tuple[bytes], int],
        pair: Tuple[bytes, bytes]
) -> Tuple[Dict[Tuple[bytes], int], Dict]:
    new_work_counter = Counter()
    new_pair_counts = defaultdict(int)

    for token, freq in word_counter.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(token[i] + token[i + 1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_work_counter[new_token] += freq

        for i in range(len(new_token) - 1):
            new_pair = (new_token[i], new_token[i + 1])
            new_pair_counts += freq

    return new_work_counter, new_pair_counts


def check_and_convert_special_tokens(special_tokens: List[str] | List[bytes]) -> List[bytes]:
    if not all(isinstance(token, bytes) for token in special_tokens):
        special_tokens_bytes = [
            token.encode('utf-8') for token in special_tokens if isinstance(token, str)
        ]
        return special_tokens_bytes

    def get_chunk_boundaries(
            file: BinaryIO,
            num_chunks: int,
            split_special_token: bytes
    ) -> List[int]:
        assert isinstance(split_special_token, bytes), 'spilt_special_toke must be bytes'

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks

        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        block_size = 4096
        for i in range(1, len(chunk_boundaries) - 1):
            pos = chunk_boundaries[i]
            file.seek(pos)
            while True:
                block = file.read(block_size)

                # EOF
                if block == b"":
                    chunk_boundaries[i] = file_size
                    break

                split_special_token_pos = block.find(split_special_token)
                if split_special_token_pos != -1:
                    chunk_boundaries[i] = pos + split_special_token_pos
                    break
                pos += block_size

        return sorted(set(chunk_boundaries))

    def train_bpe(
            input_path: str | os.PathLike,
            vocab_size: int = 10_000,
            special_tokens: List[str] = [],
            **kwargs
    ):
        special_tokens = check_and_convert_special_tokens(special_tokens)

        vocab = init_vocab(special_tokens)
        vocab_inv = {v: k for k, v in vocab.items()}
        merges: List[Tuple[bytes, bytes]] = []

        with open(input_path, mode='rb') as f:
            chunk_boundaries = get_chunk_boundaries(
                f, kwargs.get('num_processes', 8), special_tokens[0]
            )

        manager = Manager()
        queue = manager.Queue()
        processes = []

        for start, end in zip(chunk_boundaries[: -1], chunk_boundaries[1:]):
            p = Process(
                target=pre_tokenize_string_worker,
                args=(input_path, special_tokens, queue, start, end, False)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        word_counter = Counter()
        for _ in range(len(processes)):
            try:
                word_counter += queue.get(timeout=10)
            except Empty:
                print('🥵 Warning: A subprocess did not return a result!')

        pairs_freq = pair_counts(word_counter)

        num_merges = vocab_size - len(vocab)
        for _ in trange(num_merges):
            most_common_pair = get_max_freq_pair(pairs_freq)

            new_index = add_pair_to_vocab(vocab, most_common_pair, vocab_inv)
            merges.append(most_common_pair)

            word_counter, pairs_freq = merge_pair(word_counter, most_common_pair)

        return vocab, merges


class Tokenizer:
    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: Optional[List[str]] = None
    ):

        self.vocab = vocab
        self.merges = merges
        self.vocab_inv = {v: k for k, v in vocab.items()}

        if special_tokens is None:
            self.special_tokens = {}
            self.bytes_special_tokens = []
        else:
            self.special_tokens = {token: i for i, token in enumerate(special_tokens, start=len(vocab))}
            self.bytes_special_tokens = [token.encode('utf-8') for token in special_tokens if isinstance(token, str)]

    def pre_tokenize(self, text) -> List[bytes]:
        parts = split_by_special_tokens(text, list(self.special_tokens.keys()))
        token_list: List[bytes] = []

        for part in parts:
            if part in self.special_tokens.keys():
                token_list.append(part.encode('utf-8'))
            else:
                tokens = re.findall(PAT, part)
                token_list.extend(word2bytes(token) for token in tokens)

        return token_list

    def encode(self, text: str) -> List[int]:
        bytes_tokens = self.pre_tokenize(text)

        token_ids = []
        for token in bytes_tokens:
            if token in self.bytes_special_tokens:
                token_ids.append([self.vocab_inv[token]])
            else:
                token_ids.append([self.vocab_inv[b] for b in token])

        for i, pre_token in enumerate(token_ids):
            for merge in self.merges:
                new_index = self.vocab_inv.get(merge[0] + merge[1], None)
                if new_index is None:
                    continue

                merged = []
                j = 0
                while j < len(pre_token):
                    if j < len(pre_token) - 1 and (self.vocab[pre_token[j]], self.vocab[pre_token[j + 1]]) == merge:
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pre_token[j])
                        j += 1

                pre_token = merged
            token_ids[i] = pre_token

        return [i for pre in token_ids for i in pre]

    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1024) -> Iterator[int]:
        """
        Encode lines of text from an iterable using buffered batching.
        This version preserves newlines by assuming the input was split with `splitlines(keepends=True)`.
        """
        batch = []
        for line in tqdm(iterable):
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                for encoded in map(self.encode, batch):
                    yield from encoded
                batch.clear()

        if batch:
            for encoded in map(self.encode, batch):
                yield from encoded

    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character

        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @classmethod
    def from_files(
            cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_path, 'rb') as vf:
            raw_vocab = pickle.load(vf)

        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                 for k, v in raw_vocab.items()}

        with open(merges_path, 'rb') as mf:
            raw_merges = pickle.load(mf)

        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)

# -*- coding: utf-8 -*-


def check_tokenizer(tokenizer):
    assert (isinstance(tokenizer, str) and tokenizer in ("char", "word")) or callable(tokenizer), \
        f"param tokenizer must be str, must be in (char, word)"
    if isinstance(tokenizer, str):
        if tokenizer == "char":
            return lambda x: [i for i in x]
        else:
            import jieba
            return lambda x: list(jieba.cut(x))
    else:
        return tokenizer

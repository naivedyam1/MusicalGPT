# femtoGPT

**A GPT model that is a prime number.**

Yesterday, Kuber Mehta made a post about putting a GPT model inside a QR code.
What's the fun in a QR code?

I trimmed his code down to about 53 lines and made a prime number from it.

This repo contains a 5,942-digit prime whose bytes are a gzip stream that decompresses to a complete, trainable GPT implementation -- in pure Python with zero dependencies.

## Quick start

```bash
python verify_prime.py
```

To quickly run the model from the prime number

```bash
python3 -c "$(python3 verify_prime.py)"
```

This takes the prime number embedded in the script, converts it to bytes, decompresses via gzip, and prints a working GPT. Copy the output to a file and run it to train a character-level language model.

## How it works

```
picoGPT.py    Kuber Mehta's pico GPT (https://github.com/Kuberwastaken/picogpt/blob/main/picogpt.py) 
       | golf
       v
femtoGPT.py      same model in ~50 lines
       | gzip compress + pad 4 bytes
       v
  big integer       interpret bytes as big-endian int
       | gmpy2.next_prime()
       v
  THE PRIME          5,942 digits, but a valid gzip stream
```

Since gzip decompressors stop reading after the first valid stream and ignore trailing bytes. Appending 4 bytes of padding after the compressed data gives enough room (~2^32 candidates) to find a nearby prime without disturbing the gzip payload.

This is the same technique Phil Carmody used in 2001 to construct an "illegal prime" encoding DeCSS. (https://en.wikipedia.org/wiki/Illegal_number)

## Reproducing

The prime-finding scripts require [gmpy2](https://gmpy2.readthedocs.io/) for fast primality testing:

```bash
pip install -r requirements.txt
```

```bash
python find_prime.py
python verify_prime.py
```

## License

[MIT](LICENSE)

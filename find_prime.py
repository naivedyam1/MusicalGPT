import gzip, zlib
import gmpy2

with open("ultra_golf.py", "rb") as f:
    src = f.read()

compressed = gzip.compress(src, mtime=0)
print(f"Source: {len(src)} bytes -> Compressed: {len(compressed)} bytes")

# Pad 4 bytes after the gzip stream (decompressors ignore trailing bytes)
base = int.from_bytes(compressed + b'\x00' * 4, 'big')
print(f"Searching for prime ({base.bit_length()} bits)...")

candidate = int(gmpy2.next_prime(gmpy2.mpz(base)))
print(f"Found prime (offset +{candidate - base})")

prime_bytes = candidate.to_bytes((candidate.bit_length() + 7) // 8, 'big')
recovered = zlib.decompress(prime_bytes, zlib.MAX_WBITS | 16)
assert recovered == src, "Decompression mismatch"
print(f"Verified: {len(prime_bytes)} bytes, {candidate.bit_length()} bits")

hex_str = prime_bytes.hex()

with open("verify_prime.py", "w") as f:
    f.write(f'import zlib\n\n')
    f.write(f'p = 0x{hex_str}\n')
    f.write(f'b = p.to_bytes((p.bit_length() + 7) // 8, "big")\n')
    f.write(f'print(zlib.decompress(b, zlib.MAX_WBITS | 16).decode())\n')

print("Wrote verify_prime.py")

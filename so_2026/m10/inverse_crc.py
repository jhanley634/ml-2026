#! /usr/bin/env python
# ruff: file-ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
import zlib

from so_2026.m10.inverse_crc_table import crc32_table

TARGET_CRC = 0x458AF795
PREFIX_CHAR_START = ord("1")
# The total desired length is 6 bytes (1 + 5 unknowns)
TOTAL_LENGTH = 6


def calc_crc(data: bytes | bytearray, *, verbose: bool = False) -> int:
    """
    Calculates standard CRC-32, as an unsigned 32-bit integer.
    """
    crc = zlib.crc32(data)
    if verbose:
        print(f"\t{crc}\t{data!r}")
    assert 0 <= crc < 2**32, crc
    return crc


def _bytes_from_crc(crc: int) -> tuple[int, int, int, int]:
    """Return the 4 little-endian bytes of a CRC value."""
    return crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF, (crc >> 24) & 0xFF


def find_matching_crc(
    target_crc: int,
    msg: bytearray,
) -> bytearray:
    """
    Mutates msg in-place, giving it a suffix that makes the CRC match a target value.

    The final four bytes of `msg` are replaced with bytes that make the CRC work out.
    """
    if len(msg) <= 4:
        raise ValueError("msg must be longer than 4 bytes")

    prefix_bytes = bytes(msg[:-4])
    existing_crc = calc_crc(prefix_bytes)

    a0, a1, a2, a3 = _bytes_from_crc(existing_crc ^ 0xFFFFFFFF)
    f0, f1, f2, f3 = _bytes_from_crc(target_crc ^ 0xFFFFFFFF)

    e_index = next(
        i for i in range(256) if (crc32_table[i] & 0xFF000000) == (target_crc & 0xFF000000)
    )
    e0 = crc32_table[e_index] & 0xFF
    e1 = (crc32_table[e_index] >> 8) & 0xFF
    e2 = (crc32_table[e_index] >> 16) & 0xFF
    # (crc32_table[e_index] >> 24) & 0xFF  # not needed

    # d_index – third byte
    d3 = f2 ^ e2
    d_index = next(i for i in range(256) if (crc32_table[i] & 0xFF000000) == (d3 << 24))
    d0 = crc32_table[d_index] & 0xFF
    d1 = (crc32_table[d_index] >> 8) & 0xFF
    d2 = (crc32_table[d_index] >> 16) & 0xFF
    d3 = (crc32_table[d_index] >> 24) & 0xFF

    # c_index – second byte
    c3 = f1 ^ e1 ^ d2
    c_index = next(i for i in range(256) if (crc32_table[i] & 0xFF000000) == (c3 << 24))
    c0 = crc32_table[c_index] & 0xFF
    c1 = (crc32_table[c_index] >> 8) & 0xFF
    c2 = (crc32_table[c_index] >> 16) & 0xFF
    c3 = (crc32_table[c_index] >> 24) & 0xFF

    # b_index – first byte
    b3 = f0 ^ e0 ^ d1 ^ c2
    b_index = next(i for i in range(256) if (crc32_table[i] & 0xFF000000) == (b3 << 24))
    b0 = crc32_table[b_index] & 0xFF
    b1 = (crc32_table[b_index] >> 8) & 0xFF
    b2 = (crc32_table[b_index] >> 16) & 0xFF
    b3 = (crc32_table[b_index] >> 24) & 0xFF

    table_idx = [b_index, c_index, d_index, e_index]

    # Start with the inverted CRC of the prefix
    crc = existing_crc ^ 0xFFFFFFFF

    patch = bytearray(4)
    for i in range(4):
        patch[i] = (crc ^ table_idx[i]) & 0xFF
        crc = (crc >> 8) ^ crc32_table[patch[i] ^ (crc & 0xFF)]

    msg[-4:] = patch
    assert target_crc == calc_crc(msg), (target_crc, calc_crc(msg), msg)
    assert prefix_bytes == bytes(msg[:-4])
    return msg


def main() -> None: ...


if __name__ == "__main__":
    main()

#! /usr/bin/env python

import zlib

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


def find_matching_crc(
    prefix: int,
    target_crc: int,
    msg: bytearray,
) -> bytearray:
    """
    Mutates msg in-place, giving it a suffix that makes the CRC match a target value.
    """
    prefix_bytes = bytes(msg[:prefix])

    ...

    assert prefix_bytes == bytes(msg[:prefix])
    assert target_crc == calc_crc(msg)
    return msg


def main() -> None: ...


if __name__ == "__main__":
    main()

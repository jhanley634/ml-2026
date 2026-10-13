#! /usr/bin/env python

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


def find_matching_crc_initial_sketch(
    target_crc: int,
    msg: bytearray,
) -> bytearray:
    """
    Mutates msg in-place, giving it a suffix that makes the CRC match a target value.

    The final four bytes of `msg` are replaced with bytes that make the CRC work out.
    """
    assert len(msg) > 4
    prefix_bytes = bytes(msg[:-4])
    print(prefix_bytes)
    for i in range(4):
        msg[-i] = 0

    # ...

    assert prefix_bytes == bytes(msg[:-4])
    assert target_crc == calc_crc(msg)
    return msg


def find_reverse_crc(target_crc: int, known_crc: int) -> bytes:

    required_suffix_crc = target_crc ^ known_crc

    crc32 = Crc32()
    crc32.crc_ = known_crc
    return crc32.find_reverse(required_suffix_crc)


def find_matching_crc(target_crc: int, msg: bytearray) -> bytearray:
    assert len(msg) > 4
    prefix_bytes = bytes(msg[:-4])
    known_crc = calc_crc(prefix_bytes)
    suffix_bytes = find_reverse_crc(target_crc, known_crc)
    msg[-4:] = suffix_bytes

    assert target_crc == calc_crc(msg), (target_crc, calc_crc(msg), msg)

    return msg


class Crc32:
    def __init__(self) -> None:
        self.crc_ = 0

    def append(self, value: int) -> None:
        self.crc_ = (self.crc_ >> 8) ^ crc32_table[value ^ (self.crc_ & 0xFF)]

    def find_reverse(self, desired_crc: int) -> bytes:
        """
        Find a 4-byte patch that, when appended to the current CRC, produces
        ``desired_crc``.
        """
        # Build a reverse lookup map: high byte -> first table index.
        reverse_table = {}
        for i in range(len(crc32_table)):
            high_byte = (crc32_table[i] >> 24) & 0xFF
            if high_byte not in reverse_table:
                reverse_table[high_byte] = i

        crc_idx = [0] * 4
        iter_crc = ~desired_crc

        for j in range(3, -1, -1):
            target_high_byte = (iter_crc >> 24) & 0xFF
            crc_idx[j] = reverse_table.get(target_high_byte, 0)
            iter_crc = (iter_crc ^ crc32_table[crc_idx[j]]) << 8

        crc = ~self.crc_
        patch_bytes = bytearray(4)

        for j in range(4):
            patch_bytes[j] = (crc ^ crc_idx[j]) & 0xFF
            crc = (crc >> 8) ^ crc32_table[patch_bytes[j] ^ (crc & 0xFF)]

        return bytes(patch_bytes)


def main() -> None: ...


if __name__ == "__main__":
    main()

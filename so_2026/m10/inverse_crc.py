#! /usr/bin/env python
# zruff: file-ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
import zlib


def _gen_revtable(poly: int) -> list[int]:
    """
    Generate a reversed CRC table for a given polynomial.
    The algorithm matches the one used in the Julia module.
    """
    table: list[int] = [0] * 256
    for index in range(256):
        crc = index << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc ^ poly) << 1) + 1
            else:
                crc <<= 1
        table[index] = crc & 0xFFFFFFFF
    return table


# Pre-compute the reversed table for the standard CRC-32 polynomial.
REV_TABLE32 = _gen_revtable(0xEDB88320)

MASK32 = 0xFFFFFFFF


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
    # Algorithm from Stigge et al.
    if len(msg) <= 4:
        raise ValueError  # msg must be longer than 4 bytes

    prefix_bytes = bytes(msg[:-4])
    existing_crc = calc_crc(prefix_bytes) & MASK32

    crc = (target_crc ^ existing_crc) ^ MASK32
    patch = [0] * 4
    temp_crc = crc & MASK32

    for i in range(4):
        byte_val = temp_crc & 0xFF
        table_entry = REV_TABLE32[byte_val]
        patch[i] = byte_val
        temp_crc = ((temp_crc >> 8) ^ table_entry) & MASK32

    msg[-4:] = bytes(patch)

    return msg


def main() -> None: ...


if __name__ == "__main__":
    main()

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
crc32_table = REV_TABLE32


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
        raise ValueError  # msg must be longer than 4 bytes

    prefix_bytes = bytes(msg[:-4])
    existing_crc = calc_crc(prefix_bytes)

    existing_inv = existing_crc ^ 0xFFFFFFFF
    target_inv = target_crc ^ 0xFFFFFFFF

    a3 = (existing_inv >> 24) & 0xFF
    a2 = (existing_inv >> 16) & 0xFF
    a1 = (existing_inv >> 8) & 0xFF
    a0 = existing_inv & 0xFF

    f3 = (target_inv >> 24) & 0xFF
    f2 = (target_inv >> 16) & 0xFF
    f1 = (target_inv >> 8) & 0xFF
    f0 = target_inv & 0xFF

    e_index = next((i for i, val in enumerate(crc32_table) if (val >> 24) & 0xFF == f3), None)
    if e_index is None:
        raise ValueError(hex(f3))  # could not find table entry for f3 byte {f3:#04x}
    e_entry = crc32_table[e_index]
    (e_entry >> 24) & 0xFF
    e2 = (e_entry >> 16) & 0xFF
    e1 = (e_entry >> 8) & 0xFF
    e0 = e_entry & 0xFF

    d3 = f2 ^ e2
    d_index = next((i for i, val in enumerate(crc32_table) if (val >> 24) & 0xFF == d3), None)
    if d_index is None:
        raise ValueError  # could not find table entry for d3 byte {d3:#04x}
    d_entry = crc32_table[d_index]
    (d_entry >> 24) & 0xFF
    d2 = (d_entry >> 16) & 0xFF
    d1 = (d_entry >> 8) & 0xFF
    d0 = d_entry & 0xFF

    c3 = f1 ^ e1 ^ d2
    c_index = next((i for i, val in enumerate(crc32_table) if (val >> 24) & 0xFF == c3), None)
    if c_index is None:
        raise ValueError  # could not find table entry  for c3 byte {c3:#04x}
    c_entry = crc32_table[c_index]
    (c_entry >> 24) & 0xFF
    c2 = (c_entry >> 16) & 0xFF
    c1 = (c_entry >> 8) & 0xFF
    c0 = c_entry & 0xFF

    b3 = f0 ^ e0 ^ d1 ^ c2
    b_index = next((i for i, val in enumerate(crc32_table) if (val >> 24) & 0xFF == b3), None)
    if b_index is None:
        raise ValueError  # could not find table entry for b3 byte {b3:#04x}
    b_entry = crc32_table[b_index]
    (b_entry >> 24) & 0xFF
    b2 = (b_entry >> 16) & 0xFF
    b1 = (b_entry >> 8) & 0xFF
    b0 = b_entry & 0xFF

    # Compute the four patch bytes.
    x = b_index ^ a0
    y = c_index ^ b0 ^ a1
    z = d_index ^ c0 ^ b1 ^ a2
    w = e_index ^ d0 ^ c1 ^ b2 ^ a3

    patch = bytearray([x, y, z, w])
    msg[-4:] = patch

    assert target_crc == calc_crc(msg), (target_crc, calc_crc(msg), msg)
    assert prefix_bytes == bytes(msg[:-4]), "prefix was altered during patching"

    return msg


def main() -> None: ...


if __name__ == "__main__":
    main()

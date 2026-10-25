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

    # Algorithm from Stigge et al. - compute backwards CRC to find patch bytes

    # The correct approach:
    # We want to compute backwards CRC on a 4-byte buffer such that
    # when this is appended to the prefix, the total CRC equals target_crc

    # Let's compute backwards CRC with initial value = (target_crc XOR existing_crc)
    # This is the key part - we're computing what patch bytes should be
    # such that when they are CRC'd together with the prefix, we get target_crc

    # The backwards computation:
    crc = existing_crc ^ 0xFFFFFFFF  # invert initial CRC
    patch = [0] * 4

    for i in range(4):
        byte_val = crc & 0xFF
        table_entry = REV_TABLE32[byte_val]
        patch[i] = byte_val
        crc = (crc >> 8) ^ table_entry

    # The problem is that I'm not following the right algorithm exactly.
    # Let me look at the Julia code again and trace what it actually does:

    # Looking at the Julia code more carefully, it computes:
    # store_le!(a, fixpos, crc(a[firstindex(a):fixpos-1]) ⊻ 0xffffffff)
    # store_le!(a, fixpos, bwcrc(a[fixpos:end], wantcrc, revtable(crc)))

    # In our case, we're appending to the end (fixpos = length(msg) - 3)
    # But there's a subtlety in how this works.

    # The algorithm needs to be:
    # 1. Compute CRC of prefix
    # 2. Compute what patch bytes should be using backwards CRC computation

    # Let's try the correct approach:
    # We compute backwards CRC where we know the final result should be target_crc
    # and we work backwards to find what 4 bytes would produce this

    # Correct backwards computation for our case:
    crc = target_crc ^ 0xFFFFFFFF  # Start with inverted target CRC

    patch = [0] * 4
    temp_crc = crc

    for i in range(4):
        byte_val = temp_crc & 0xFF
        table_entry = REV_TABLE32[byte_val]
        patch[i] = byte_val
        temp_crc = (temp_crc >> 8) ^ table_entry

    # Apply the patch to message
    msg[-4:] = bytes(patch)

    # Verify it works
    assert target_crc == calc_crc(msg), (target_crc, calc_crc(msg), msg)
    assert prefix_bytes == bytes(msg[:-4]), "prefix was altered during patching"

    return msg


def main() -> None: ...


if __name__ == "__main__":
    main()

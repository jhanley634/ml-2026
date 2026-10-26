import zlib

MASK32 = 0xFFFF_FFFF


def _gen_revtable(poly: int) -> list[int]:
    """
    Generate a reversed CRC table for a given polynomial.
    The algorithm matches the one used in the Julia module.
    """
    table: list[int] = [0] * 256
    for index in range(256):
        crc = index << 24
        for _ in range(8):
            if crc & 0x8000_0000:
                crc = ((crc ^ poly) << 1) | 1
            else:
                crc <<= 1
            crc &= MASK32

        table[index] = crc

    return table


# Pre-compute the reversed table for the standard CRC-32 polynomial.
_REV_TABLE32 = _gen_revtable(0xEDB88320)


def calc_crc(data: bytes | bytearray, *, verbose: bool = False) -> int:
    """
    Calculates standard CRC-32, as an unsigned 32-bit integer.
    """
    crc = zlib.crc32(data)
    if verbose:
        print(f"\t{crc}\t{data!r}")
    assert 0 <= crc < 2**32, crc
    return crc


def bytes_from_crc(crc: int) -> tuple[int, int, int, int]:
    """Return the 4 little-endian bytes of a CRC value."""
    return crc & 0xFF, (crc >> 8) & 0xFF, (crc >> 16) & 0xFF, (crc >> 24) & 0xFF


def bwcrc(a: bytes | bytearray, crc: int) -> int:
    """Backward CRC calculation used by the inverse CRC algorithm."""
    crc ^= MASK32
    for i in range(len(a) - 1, -1, -1):
        byte_val = a[i]
        crc = ((crc << 8) ^ _REV_TABLE32[(crc >> 24) & 0xFF] ^ byte_val) & MASK32
    return crc


def find_matching_crc(
    target_crc: int,
    msg: bytearray,
) -> bytearray:
    """
    Mutates msg in-place, replacing its last 4 bytes with values that make
    the CRC of the entire message equal to target_crc.
    """
    # Algorithm from Stigge et al.
    if len(msg) < 4:
        raise ValueError  # msg must be longer than 4 bytes

    prefix = msg[:-4]
    prefix_crc = calc_crc(prefix) ^ MASK32
    msg[-4:] = bytes_from_crc(prefix_crc)

    suffix = msg[-4:]
    new_crc = bwcrc(suffix, target_crc)
    msg[-4:] = bytes_from_crc(new_crc)

    return msg

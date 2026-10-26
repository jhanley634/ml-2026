from so_2026.m10.inverse_crc import MASK32, bwcrc, bytes_from_crc, calc_crc


def find_matching_crc_at_offset(
    target_crc: int,
    msg: bytearray,
    offset: int,
) -> bytearray:
    """
    Mutates msg in-place, replacing the four bytes at the specified offset
    with values that make the CRC of the message equal the desired target_crc.
    """
    if len(msg) < offset + 4:
        raise ValueError

    prefix_crc = calc_crc(msg[:offset]) ^ MASK32

    msg[offset : offset + 4] = bytes_from_crc(prefix_crc)

    msg[offset : offset + 4] = bytes_from_crc(bwcrc(msg[offset:], target_crc))

    return msg

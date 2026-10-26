import unittest

from so_2026.m10.inverse_crc import bwcrc, bytes_from_crc, calc_crc, find_matching_crc
from so_2026.m10.inverse_crc_at_offset import find_matching_crc_at_offset

M0 = b"0hello"  # this is message M0
M0_CRC = 1_166_735_253

MASK32 = 0xFFFF_FFFF


class InverseCrcTest(unittest.TestCase):

    def test_crc_calculate(self) -> None:
        msg = bytearray(M0)
        self.assertEqual(M0_CRC, calc_crc(msg))

        for i in range(10):
            msg[0] = ord("0") + i
            self.assertGreaterEqual(calc_crc(msg), 0)

    def test_bwcrc(self) -> None:
        msg = bytearray(b"hello")
        crc = calc_crc(msg)
        backward_crc = bwcrc(msg, crc)
        self.assertEqual(907_060_870, crc)
        self.assertEqual(4_294_967_295, backward_crc)

        # Test with a message that has 4 bytes and check if we can adjust it
        msg_with_padding = bytearray(b"test") + bytes([0, 0, 0, 0])
        original_crc = calc_crc(msg_with_padding[:-4])
        msg_with_padding[-4:] = bytes_from_crc(original_crc ^ MASK32)

        target_crc = 0x12345678
        new_crc = bwcrc(msg_with_padding[-4:], target_crc)
        msg_with_padding[-4:] = bytes_from_crc(new_crc)
        final_crc = calc_crc(msg_with_padding)
        self.assertEqual(target_crc, final_crc)

    def test_find_matching_crc(self) -> None:
        m1 = bytearray(M0)
        m1[0] = ord("1")  # "1hello"
        m2 = find_matching_crc(M0_CRC, m1)
        self.assertStartsWith(m2.decode(), "1h")
        self.assertEqual(M0_CRC, calc_crc(m2))

    def test_find_matching_crc_property_based(self) -> None:
        msg = b"'Twas brillig and the slithey toves did gyre and gimble in the wabe."
        expected = calc_crc(msg)
        while len(msg) > 4:
            new_msg = find_matching_crc(expected, bytearray(msg))
            self.assertEqual(expected, calc_crc(new_msg))
            msg = msg[:-1]

    def test_find_matching_crc_at_offset(self, *, verbose: bool = False) -> None:
        msg = b"'Twas brillig and the slithey toves did gyre and gimble in the wabe"
        expected = calc_crc(msg + b"?")
        for i in range(len(msg) - 3):
            new_msg = find_matching_crc_at_offset(expected, bytearray(msg), i)
            self.assertNotEqual(new_msg, msg)
            if verbose:
                print(i, new_msg)
            self.assertEqual(expected, calc_crc(new_msg))

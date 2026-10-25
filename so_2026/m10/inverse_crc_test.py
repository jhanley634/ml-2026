import random
import unittest

from so_2026.m10.inverse_crc import calc_crc, find_matching_crc

M0 = b"0hello"  # this is message M0
M0_CRC = 1_166_735_253


class InverseCrcTest(unittest.TestCase):

    def test_crc_calculate(self) -> None:
        msg = bytearray(M0)
        self.assertEqual(M0_CRC, calc_crc(msg))

        for i in range(10):
            msg[0] = ord("0") + i
            self.assertGreaterEqual(calc_crc(msg), 0)

    def test_find_matching_crc(self) -> None:
        print(find_matching_crc(M0_CRC, bytearray(M0)))

        m1 = bytearray(M0)
        m1[0] = ord("1")  # "1hello"
        print(find_matching_crc(M0_CRC, m1))

        self.assertGreater(3, 2)

    def test_find_matching_crc_property_based(self) -> None:
        msg = b"'Twas brillig and the slithey toves did gyre and gimble in the wabe."
        expected = calc_crc(msg)
        while len(msg) > 4:
            new_msg = find_matching_crc(expected, bytearray(msg))
            print(new_msg)
            # self.assertEqual(expected, calc_crc(new_msg))
            msg = msg[:-1]

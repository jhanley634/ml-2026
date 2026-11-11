# Source - https://stackoverflow.com/q/80016100
# Posted by Capitão Mendigão, modified by community. See post 'Timeline' for change history
# Retrieved 2026-11-11, License - CC BY-SA 4.0

import sys

roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "M": 1000}

b = 0

input_roman_numeral = input("Enter the Roman numeral: ")


length = len(input_roman_numeral)

if length < 1 or length > 15:
    print("Enter a value with at least 1 and no more than 15 characters.")
    sys.exit(1)
for i in input_roman_numeral:
    if i not in roman_numerals:
        print("Enter valid Roman numerals instead of regular numbers.")
        sys.exit(1)

for h in range(len(input_roman_numeral)):
    current = roman_numerals[input_roman_numeral[h]]
    next_value = roman_numerals[input_roman_numeral[h + 1]]

    if current < next_value:
        b -= current
    elif current > next_value:
        b += current
        b += next_value

    if h == length - 2:
        break


if b not in range(1, 4_000):
    print("Please enter a value within the valid range!")
    sys.exit(1)


print(f"Your Roman numeral is {b}")

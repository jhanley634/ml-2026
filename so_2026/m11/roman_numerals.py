# Source - https://stackoverflow.com/q/80016100
# Posted by Capitão Mendigão, modified by community. See post 'Timeline' for change history
# Retrieved 2026-11-11, License - CC BY-SA 4.0


roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_decimal(roman: str) -> int:
    b = 0

    if len(roman) not in range(1, 16):
        msg = f"wrong length, must be at most fifteen: {len(roman)}"
        raise ValueError(msg)
    for ch in roman:
        if ch not in roman_values:
            msg = f"not a valid Roman numeral, {ch=}"
            raise ValueError(msg)

    for h in range(len(roman) - 1):
        current = roman_values[roman[h]]
        next_value = roman_values[roman[h + 1]]

        if current < next_value:
            b -= current
        elif current > next_value:
            b += current
            b += next_value

    if b not in range(1, 4_000):
        raise ValueError(b)

    return b


if __name__ == "__main__":
    dec = roman_to_decimal(input("Enter the Roman numeral: "))
    print(f"Your Roman numeral is {dec}")

# Source - https://codereview.stackexchange.com/q/252593
# Posted by Martins Micheal, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-18, License - CC BY-SA 4.0

from decimal import Decimal
from math import factorial, floor, sqrt


def is_prime(x):
    if x == 2 or x == 3 or x == 5:
        return True
    if (x % 2 == 0) or (x % 3 == 0) or (x % 5 == 0) or (x == 1):
        return False
    for i in range(6, floor(sqrt(x)), 6):
        if x % (i + 1) == 0:
            return False
        elif i % (i + 5) == 0:
            return False
    return True


def decomp(n):
    y = factorial(n)
    if y < 7:
        pri = list(filter(is_prime, range(1, y + 1)))
    else:
        pri = list(filter(is_prime, range(1, floor(Decimal(y).sqrt() + 1))))
    a = {}
    b = 0
    for i in pri:
        while y % i == 0:
            y = y // i
            b += 1
            a[i] = b
            b = 0
    if len(list(a.keys())) == 1:
        if list(a.values()) != [1]:

            c = str(a.keys()) + "^" + str(a.values())
        else:
            c = str(list(a.keys()))
    elif len(list(a.keys())) > 1:
        if a[list(a.keys())[0]] > 1:
            c = str(list(a.keys())[0]) + "^" + str(a[list(a.keys())[0]])

        elif a[list(a.keys())[0]] == 1:
            c = str(list(a.keys())[0])
        for key in list(a.keys())[1:]:
            if a[key] != 1:
                c += " * " + str(key) + "^" + str(a[key])

            elif a[key] == 1:
                c += " * " + str(key)

    return c

# from https://codereview.stackexchange.com/a/301010/find-median-of-two-sorted-arrays


def merge(lst1, lst2):
    """
    Creates a generator yielding sorted,
    merged contents of two input lists.
    """

    i, j = 0, 0

    while i < len(lst1) or j < len(lst2):
        if i >= len(lst1) or lst1[i] >= lst2[j]:
            yield lst2[j]
            j += 1
        else:
            yield lst1[i]
            i += 1

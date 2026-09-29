import random

TEST_SIZE = 1000
DEV_SIZE = 200


def shuffled(items):
    items = list(items)
    random.Random(13).shuffle(items)
    return items


def split(test, dev=None, rest=(), test_size=TEST_SIZE):
    test = shuffled(test)
    if dev is None:
        dev, test = test[: len(test) // 10], test[len(test) // 10 :]
    dev = shuffled(dev)
    return {"dev": dev[:DEV_SIZE], "test": test[:test_size], "rest": dev[DEV_SIZE:] + test[test_size:] + list(rest)}

#!/usr/bin/env python

import sys


def main():
    result = []
    for arg in sys.argv[1:]:
        prefix = arg[: arg.rfind("/")]
        num = int(arg[arg.rfind("/") + 1 :])
        result.extend(f"{prefix}/{i}" for i in range(num))

    print(",".join(result))


if __name__ == "__main__":
    main()

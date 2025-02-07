"""
str intermediate Decimal: 	0.0005830000009154901 ms.
str intermediate float: 	0.00043300002289470285 ms.
round_half_even Decimal: 	0.0015780000103404745 ms.
round_half_even float: 		0.0004810000245925039 ms.
"""

import timeit
from decimal import Decimal

from polypy.rounding import round_half_even

nb_iter = 100
x = 0.01


def main():  # sourcery skip: remove-unnecessary-cast
    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        Decimal(str(x))
    end_t = timeit.default_timer()
    print(f"str intermediate Decimal: \t{((end_t - start_t)/nb_iter)*1_000} ms.")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        float(str(x))
    end_t = timeit.default_timer()
    print(f"str intermediate float: \t{((end_t - start_t)/nb_iter)*1_000} ms.")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        round_half_even(Decimal(x), 2)
    end_t = timeit.default_timer()
    print(f"round_half_even Decimal: \t{((end_t - start_t) / nb_iter) * 1_000} ms.")

    start_t = timeit.default_timer()
    for _ in range(nb_iter):
        round_half_even(float(x), 2)
    end_t = timeit.default_timer()
    print(f"round_half_even float: \t\t{((end_t - start_t) / nb_iter) * 1_000} ms.")


if __name__ == "__main__":
    main()

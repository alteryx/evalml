import numpy as np


def assert_params_almost_equal(a, b, decimal=7):
    """Given two sets of numeric/str parameter lists, assert numerics are approx equal and strs are equal"""
    def separate_numeric_and_str(values):
        def is_numeric(val):
            return isinstance(val, (int, float))

        def extract(vals, invert):
            return [el for el in vals if (invert ^ is_numeric(el))]

        return extract(values, False), extract(values, True)
    a_num, a_str = separate_numeric_and_str(a)
    b_num, b_str = separate_numeric_and_str(a)
    assert a_str == b_str
    np.testing.assert_almost_equal(a_num, b_num, decimal=decimal,
                                   err_msg="Numeric parameter values are not approximately equal")

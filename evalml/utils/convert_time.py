def convert_to_seconds(input_str):
    hours = {'h', 'hour', 'hr'}
    minutes = {'m', 'minute', 'min'}
    seconds = {'s', 'second', 'sec'}
    value, unit = input_str.split()
    if unit[-1] == 's':
        unit = unit[:-1]
    if unit in seconds:
        return float(value)
    elif unit in minutes:
        return float(value) * 60
    elif unit in hours:
        return float(value) * 3600
    else:
        msg = "Invalid unit. Units must be hours, mins, or seconds. Received '{}'".format(unit)
        raise AssertionError(msg)

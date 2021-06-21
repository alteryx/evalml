from pkg_resources import iter_entry_points

for entry_point in iter_entry_points("alteryx_open_src_initialize"):
    try:
        method = entry_point.load()
        if callable(method):
            method("evalml")
    except Exception:
        pass

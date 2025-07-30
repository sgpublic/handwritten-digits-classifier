def _fmt(precision: int = 6):
    return f"{{:.{precision}g}}"

def float2str(f: float, precision: int = 6) -> str:
    return _fmt(precision).format(f)

def arr2str(arr: list[float], precision: int = 6) -> str:
    return f"[{', '.join(float2str(x) for x in arr)}]"
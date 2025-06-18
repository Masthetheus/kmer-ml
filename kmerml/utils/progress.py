import time

def progress_bar(current, total, start_time=None, bar_length=30):
    if start_time is None:
        start_time = time.time()
    elapsed = time.time() - start_time
    proportion = current / total
    complete = int(proportion * bar_length)
    bar = "|" + "o" * complete + "-" * (bar_length - complete) + "|"
    if current > 0:
        mean_time = elapsed / current
        restante = mean_time * (total - current)
    else:
        restante = 0
    minutes, seconds = divmod(int(restante), 60)
    print(f"\r{bar} {current}/{total} - Expected completion in: {minutes:02d}:{seconds:02d}", end='', flush=True)
    if current == total:
        print()
    return start_time
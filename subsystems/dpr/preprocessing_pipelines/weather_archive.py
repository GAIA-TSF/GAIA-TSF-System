"""Bounded weather requests, including retries for streamed API timeouts."""
from contextlib import nullcontext
import time

import pandas as pd
from requests.exceptions import ConnectionError, Timeout
from openmeteo_requests.Client import OpenMeteoRequestsError


def fetch_weather_frame(client, session, url, params, timezone, logger, attempts=3):
    """Fetch inclusive monthly slices; concatenate before downstream rolling metrics."""
    if attempts < 1:
        raise ValueError('Weather attempts must be at least one')
    start, end = pd.Timestamp(params['start_date']), pd.Timestamp(params['end_date'])
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('Weather date bounds must be valid and ordered')
    daily = 'daily' in params
    variables = params['daily' if daily else 'hourly']
    frames = []
    while start <= end:
        stop = min(start + pd.offsets.MonthEnd(0), end)
        chunk = dict(params, start_date=start.date().isoformat(), end_date=stop.date().isoformat())
        logger.info(f"Weather request: {url} {chunk['start_date']} through {chunk['end_date']}")
        for attempt in range(attempts):
            try:
                # A streamed error can have HTTP 200 and therefore be cached.
                # Bypass that response on retries rather than replaying it.
                with session.cache_disabled() if attempt else nullcontext():
                    response = client.weather_api(url, params=chunk, timeout=(15, 120))[0]
                break
            except (OpenMeteoRequestsError, ConnectionError, Timeout) as exc:
                message = str(exc).lower()
                minute_limit = 'minutely api request limit exceeded' in message
                transient = minute_limit or isinstance(exc, (ConnectionError, Timeout)) or any(
                    token in message for token in ('timeout', 'timed out', 'temporarily', '502', '503', '504'))
                if not transient or attempt == attempts - 1:
                    raise RuntimeError(
                        f"Weather request failed for {chunk['start_date']} through {chunk['end_date']} "
                        f"at ({params['latitude']}, {params['longitude']}): {exc}"
                    ) from exc
                # The API asks for one minute. Do not apply short timeout backoff
                # to quotas, or retry hourly/daily limits as minute limits.
                delay = 60 if minute_limit else 2 ** (attempt + 1)
                logger.warning(f'Weather request failed; retry {attempt + 2}/{attempts} in {delay}s: {exc}')
                time.sleep(delay)
        values = response.Daily() if daily else response.Hourly()
        dates = pd.date_range(
            start=pd.to_datetime(values.Time(), unit='s', utc=True).tz_convert(timezone).tz_localize(None),
            periods=len(values.Variables(0).ValuesAsNumpy()), freq='D' if daily else 'h')
        frame = pd.DataFrame({'date' if daily else 'datetime': dates})
        for index, name in enumerate(variables):
            frame[name] = values.Variables(index).ValuesAsNumpy()
        frames.append(frame)
        start = stop + pd.Timedelta(days=1)
    return pd.concat(frames, ignore_index=True)


class WeatherStore:
    """Cache decoded monthly frames and serialize requests across local processes.

    All callers sharing root use the same advisory lock and request clock.
    Only historical slices ending at least 30 days ago are cached indefinitely.
    """

    def __init__(self, root, interval=2.0):
        from pathlib import Path
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval = interval

    def fetch(self, client, session, url, params, timezone, logger):
        import fcntl
        import hashlib
        import json
        from types import SimpleNamespace

        frames = []
        start, end = pd.Timestamp(params['start_date']), pd.Timestamp(params['end_date'])
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError('Weather date bounds must be valid and ordered')
        date_column = 'date' if 'daily' in params else 'datetime'
        clock_path = self.root / 'last_request.txt'

        def paced_request(*args, **kwargs):
            # Called under the shared lock, including retries and cooldowns.
            last = float(clock_path.read_text()) if clock_path.exists() else 0
            delay = max(0, self.interval - (time.time() - last))
            if delay:
                time.sleep(delay)
            clock_path.write_text(str(time.time()))
            return client.weather_api(*args, **kwargs)

        while start <= end:
            stop = min(start + pd.offsets.MonthEnd(0), end)
            chunk = dict(params, start_date=start.date().isoformat(), end_date=stop.date().isoformat())
            # Normalize insignificant coordinate rounding, including the grid centre.
            for key in ('latitude', 'longitude'):
                chunk[key] = round(float(chunk[key]), 8)
            identity = json.dumps({'schema': 1, 'url': url, 'params': chunk, 'timezone': timezone}, sort_keys=True)
            key = hashlib.sha256(identity.encode()).hexdigest()
            path = self.root / f'{key}.csv'
            with (self.root / 'requests.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                historical = stop < pd.Timestamp.now(tz='UTC').tz_localize(None).normalize() - pd.Timedelta(days=30)
                fresh = path.exists() and (historical or time.time() - path.stat().st_mtime < 3600)
                if fresh:
                    logger.info(f'Weather cache hit: {chunk["start_date"]} through {chunk["end_date"]}')
                    frame = pd.read_csv(path, parse_dates=[date_column], float_precision='round_trip')
                else:
                    # Decoded data are the authoritative cache; never cache a streamed error.
                    with session.cache_disabled():
                        frame = fetch_weather_frame(SimpleNamespace(weather_api=paced_request),
                                                    session, url, chunk, timezone, logger)
                    if frame.empty:
                        raise ValueError(f'Empty weather response for {chunk}')
                    tmp = path.with_suffix('.tmp')
                    try:
                        frame.to_csv(tmp, index=False)
                        tmp.replace(path)
                    finally:
                        tmp.unlink(missing_ok=True)
                    (self.root / f'{key}.json').write_text(identity)
            frame[date_column] = frame[date_column].astype('datetime64[ns]')
            frames.append(frame)
            start = stop + pd.Timedelta(days=1)
        return pd.concat(frames, ignore_index=True)

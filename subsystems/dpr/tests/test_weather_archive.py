from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from openmeteo_requests.Client import OpenMeteoRequestsError

from subsystems.dpr.preprocessing_pipelines.weather_archive import fetch_weather_frame


def test_monthly_requests_retry_stream_timeout_and_keep_continuity(monkeypatch):
    monkeypatch.setattr('subsystems.dpr.preprocessing_pipelines.weather_archive.time.sleep', lambda _: None)
    client, session = Mock(), Mock()
    session.cache_disabled.side_effect = nullcontext
    calls = []
    def request(url, params, **kwargs):
        calls.append(params.copy())
        if len(calls) == 1:
            raise OpenMeteoRequestsError('Unexpected error while streaming data: timeoutReached')
        days = pd.date_range(params['start_date'], params['end_date'])
        values = Mock()
        values.Time.return_value = days[0].tz_localize('UTC').timestamp()
        values.Variables.return_value.ValuesAsNumpy.return_value = np.ones(len(days))
        response = Mock()
        response.Daily.return_value = values
        return [response]
    client.weather_api.side_effect = request
    frame = fetch_weather_frame(client, session, 'url', dict(start_date='2020-01-29',
        end_date='2020-03-02', daily=['rain'], latitude=0, longitude=0), 'UTC', Mock())
    assert [(c['start_date'], c['end_date']) for c in calls] == [
        ('2020-01-29', '2020-01-31'), ('2020-01-29', '2020-01-31'),
        ('2020-02-01', '2020-02-29'), ('2020-03-01', '2020-03-02')]
    assert frame.date.tolist() == list(pd.date_range('2020-01-29', '2020-03-02'))
    assert frame.rain.rolling(7).sum().iloc[-1] == 7
    session.cache_disabled.assert_called_once()


@pytest.mark.parametrize('message,count', [('timeoutReached', 3), ('Invalid variable', 1),
    ('Minutely API request limit exceeded. Please try again in one minute.', 3),
    ('Daily API request limit exceeded.', 1)])
def test_failures_are_bounded_and_not_silenced(monkeypatch, message, count):
    monkeypatch.setattr('subsystems.dpr.preprocessing_pipelines.weather_archive.time.sleep', lambda _: None)
    client, session = Mock(), Mock()
    session.cache_disabled.side_effect = nullcontext
    client.weather_api.side_effect = OpenMeteoRequestsError(message)
    with pytest.raises(RuntimeError, match='2020-01-01 through 2020-01-02'):
        fetch_weather_frame(client, session, 'url', dict(start_date='2020-01-01',
            end_date='2020-01-02', daily=['rain'], latitude=0, longitude=0), 'UTC', Mock())
    assert client.weather_api.call_count == count


def test_minute_quota_waits_then_retries_same_month(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr('subsystems.dpr.preprocessing_pipelines.weather_archive.time.sleep', sleep)
    client, session = Mock(), Mock()
    session.cache_disabled.side_effect = nullcontext
    response = Mock()
    response.Daily.return_value.Time.return_value = pd.Timestamp('2020-03-01', tz='UTC').timestamp()
    response.Daily.return_value.Variables.return_value.ValuesAsNumpy.return_value = np.ones(31)
    client.weather_api.side_effect = [
        OpenMeteoRequestsError("failed to request archive: {'reason': 'Minutely API request limit exceeded. Please try again in one minute.', 'error': True}"),
        [response],
    ]
    result = fetch_weather_frame(client, session, 'url', dict(start_date='2020-03-01',
        end_date='2020-03-31', daily=['rain'], latitude=-29.76, longitude=25.42), 'UTC', Mock())
    sleep.assert_called_once_with(60)
    assert client.weather_api.call_args_list[0] == client.weather_api.call_args_list[1]
    session.cache_disabled.assert_called_once()
    assert len(result) == 31


def test_store_reuses_decoded_data_and_paces_misses(tmp_path, monkeypatch):
    from subsystems.dpr.preprocessing_pipelines.weather_archive import WeatherStore
    now = [100.]
    waits = []
    monkeypatch.setattr('subsystems.dpr.preprocessing_pipelines.weather_archive.time.time', lambda: now[0])
    def sleep(delay):
        waits.append(delay)
        now[0] += delay
    monkeypatch.setattr('subsystems.dpr.preprocessing_pipelines.weather_archive.time.sleep', sleep)
    session, client = Mock(), Mock()
    session.cache_disabled.side_effect = nullcontext
    response = Mock()
    response.Daily.return_value.Time.return_value = pd.Timestamp('2020-03-01', tz='UTC').timestamp()
    response.Daily.return_value.Variables.return_value.ValuesAsNumpy.return_value = np.ones(31)
    client.weather_api.return_value = [response]
    params = dict(start_date='2020-03-01', end_date='2020-03-31', daily=['rain'], latitude=-29.76, longitude=25.42)
    first = WeatherStore(tmp_path).fetch(client, session, 'url', params, 'UTC', Mock())
    # A new instance represents a subsequent run; insignificant centre rounding is shared.
    second = WeatherStore(tmp_path).fetch(client, session, 'url', dict(params, latitude=-29.760000000000005), 'UTC', Mock())
    pd.testing.assert_frame_equal(first, second)
    assert client.weather_api.call_count == 1
    WeatherStore(tmp_path).fetch(client, session, 'url', dict(params, longitude=25.43), 'UTC', Mock())
    assert client.weather_api.call_count == 2
    assert waits == [2.0]


def test_store_does_not_cache_failed_request(tmp_path):
    from subsystems.dpr.preprocessing_pipelines.weather_archive import WeatherStore
    client, session = Mock(), Mock()
    session.cache_disabled.side_effect = nullcontext
    client.weather_api.side_effect = OpenMeteoRequestsError('Invalid variable')
    with pytest.raises(RuntimeError):
        WeatherStore(tmp_path).fetch(client, session, 'url', dict(start_date='2020-01-01',
            end_date='2020-01-02', daily=['rain'], latitude=0, longitude=0), 'UTC', Mock())
    assert not list(tmp_path.glob('*.csv'))

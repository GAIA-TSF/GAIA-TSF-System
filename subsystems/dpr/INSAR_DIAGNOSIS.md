
### Weather archive streaming timeouts

An Open-Meteo `timeoutReached` response during `_environmental_database` is an
external weather-data failure after InSAR export. The website being reachable
is not evidence that a particular archive query completed successfully.
`weather_archive.fetch_weather_frame()` now splits climate and air-quality
requests into inclusive calendar-month slices, uses explicit connection/read
socket timeouts, and retries transient client/stream errors up to three times
(with 2 and 4 second backoff). Retries bypass the HTTP response cache because
an HTTP-200 response may contain a streamed error. The existing HTTP transport
retry layer also remains active. Permanent API errors are not silently ignored.
Monthly frames are concatenated before rolling climate indicators are computed.
Completed climate CSV output is saved before air-quality requests begin.

These changes apply to new runs; they do not resume a failed environmental
stage automatically. Existing InSAR GeoTIFFs were exported before this stage.
For runs requiring only InSAR products, use `--insar-only` to skip environmental
and risk processing. Historical air-quality availability is separate from
weather-archive availability; unavailable periods still require explicit handling.

Minute-quota errors (`Minutely API request limit exceeded`) now wait 60 seconds
before retrying the same monthly request, bypassing cached responses on retry.
The total remains bounded at three application-level attempts per chunk. Hourly
and daily quota errors are not treated as minute limits. This handles temporary
throttling but cannot guarantee success while other clients consume a shared
quota. The change applies to newly started processes; existing InSAR exports
remain available if the environmental stage exhausts its retries.

Weather downloads now use a shared decoded-data cache at
`/data/gaia_tsf/weather_cache/` for the standard site runner. Each successful
monthly request is keyed by endpoint, parameters (including model, variables,
timezone and dates) and coordinates rounded to eight decimal places. Duplicate
centre/grid coordinates reuse the same data while retaining both output location
names. Cache files are committed atomically; failed responses are not stored.
Slices ending more than 30 days ago are retained indefinitely; newer slices
expire after one hour. Delete the corresponding CSV to force a refresh.

A shared filesystem lock serializes downloads across runs using this cache, with
at least two seconds between application-level requests. The existing minute
quota cooldown remains active, and hidden transport retries are disabled so all
retries pass through pacing. Processes on other hosts or using another cache
root are not coordinated. This reduces request bursts but cannot guarantee that
shared-IP or longer-period quotas will never be reached. Earlier HTTP caches are
not migrated; decoded caching starts on the next run. Completed months survive
later failures, but automatic restart of only the environmental stage remains
unimplemented.

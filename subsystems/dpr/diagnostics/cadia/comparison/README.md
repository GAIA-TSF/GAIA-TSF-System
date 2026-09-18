# Interpretation of corrected Cadia displacement (18 September 2026)

The corrected series contains 103 dates, 1 March 2015–22 December 2018.
Dated rasters contain cumulative LOS displacement in mm relative to the first
acquisition and the configured reference areas. Negative is away from the
satellite, not necessarily pure vertical settlement. Persistent colours do not
mean a new failure or a new displacement of that magnitude at each date.

The ±100 mm screenshot legend saturates: the whole-image first percentiles are
−110.9 mm on 12 February 2016, −204.1 mm on 9 March 2018 and −372.9 mm on
22 December 2018. Values below −100 mm cannot be distinguished by this legend.
The image labelled average velocity is a different quantity from cumulative
LOS. Its legend gives mm without an explicit time denominator, so its colours
alone are insufficient for a numerical rate comparison.

## Time and footprint

Filenames use UTC dates. The source acquisition 20180309T192342 occurred on
**10 March 2018 at 06:23:42 Australia/Sydney**, after the 9 March evening failure.
The source 25 February UTC date corresponds to 26 February locally, explaining
its difference from the date in Carlà et al. Do not interpret the March 9 UTC
product as a precursor.

Extraction used the existing `TSF_Slope_Failure_area_Cadia_Gold_AUS.gpkg` from
the project's data/static directory. This is a footprint statistic, not a
replication of an individual radar target in a paper. There are 677 raster pixel
centres within the polygon; 530 remain finite throughout all 103 dates. The
cumulative area time series uses those common pixels to avoid changing spatial
support. Raster cells are not independent observations.

| UTC date | Common-pixel median cumulative LOS, mm |
| --- | ---: |
| 2016-02-12 | -20.1 |
| 2018-01-20 | -32.6 |
| 2018-02-25 | -48.3 |
| 2018-03-09 | -49.0 |
| 2018-12-22 | -114.4 |

Pixelwise change between 20 January and 25 February 2018 over pixels finite at
both endpoints: median **−9.52 mm**, mean **−10.75 mm**, spatial 10th–90th
percentiles **−22.91 to +0.26 mm**. The median of pixelwise differences is not
necessarily the difference between the two spatial medians. Spatial spread is
not a confidence interval.

## Comparison with supplied publications

[Thomas et al. (2019), sections 2 and 3.1](https://papers.acg.uwa.edu.au/p/1910_11_Thomas/)
report 29 mm movement at a selected location from 20 January to 25 February
2018. They discuss earlier low-magnitude motion and acceleration from late
January. Their workflow includes atmospheric filtering and rejection of
inconsistent unwrapping, and separates pre- and post-collapse epochs.

[Carlà et al. (2019), Fig. 3 and Methods](https://www.nature.com/articles/s41598-019-50792-y)
use SqueeSAR target selection and a shorter monitoring window. They report up
to 29.9 mm during 14–26 February (local dates) and 40–68.9 mm at the more rapidly
moving targets over January–March. That cannot be equated with an entire
footprint's median or with broad tailings-interior colours.

There is negative change in the same general sector in the new interval map,
but neither matching a published target nor validated agreement in magnitude
or acceleration has been established. Do not tune processing to reproduce the
papers' colours. Their absence of displayed points in a region is not proof
of zero displacement there.

## Remaining quality limitations

Independent reference-area mean time series show excursions of several tens
of mm. A combined spatial reference constrains their aggregate, not each
polygon individually. This can reflect residual atmospheric/DEM/unwrapping
errors, changing support, or actual movement; this check does not identify
which cause dominates. These are reference-consistency checks, not independent
validation sites, since the areas also entered processing.

On each quality raster's own grid, the failure-footprint medians are mean
coherence 0.514, valid-pair fraction 0.913, and pair-fit RMSE 6.89 mm (spatial
10th–90th RMSE range 5.67–13.24 mm). These are whole-stack aggregate diagnostics,
not date-specific accuracy or uncertainty estimates. Low residuals do not rule
out consistent bias or incorrect unwrap components.

The current pipeline has no replacement spatial atmospheric correction after
removal of the destructive 90 m high-pass. Spatial unwrap-component consistency
and reliable behaviour across the collapse are also not established. InSAR
cannot recover collapse magnitude where the surface loses coherence. Thus,
post-failure cumulative values are not automatically a valid measurement of the
collapse displacement.

For validation, compare matched pre-failure intervals and robust time series
on the dam crest/face; verify coherence and unwrap consistency per date and
reference patch; independently process pre- and post-failure stacks; and assess
atmospheric/DEM effects using stable surrounding ground. A same-window velocity
product can then be compared with a published velocity map. No new processing
filters or thresholds were introduced for this comparison.

## Files

- `comparison.png`: cumulative and rebased footprint series, individual reference
  series, and matched-interval map. Shading denotes spatial spread.
- `area_timeseries.csv`: common-pixel time series for the failure footprint,
  whole TSF and each reference polygon.
- `displacement_20180120_20180225_mm.tif`: subtraction of existing cumulative
  rasters, not a new independent inversion or a quality-screened final product.

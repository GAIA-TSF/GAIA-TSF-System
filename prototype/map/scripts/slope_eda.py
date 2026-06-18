#!/usr/bin/env python3

import os
import glob
import json
import yaml
import numpy as np
import pandas as pd

import rasterio

import matplotlib.pyplot as plt

from scipy.stats import median_abs_deviation

from rasterio.transform import Affine


class SlopeEDA:

    def __init__(self, cfg):

        self.cfg = cfg

        self.inputs_dir = cfg["project"]["inputs_dir"]

        self.static_dir = cfg["project"]["static_dir"]

        self.output_dir = os.path.join(
            cfg["project"]["output_dir"],
            "eda"
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    # =====================================================
    # READ MASK
    # =====================================================

    def load_mask(self):

        mask_file = os.path.join(
            self.static_dir,
            self.cfg["masks"]["tsf_mask"]
        )

        with rasterio.open(mask_file) as src:

            self.mask = src.read(1) > 0

            self.profile = src.profile
        
        print(self.mask.shape)

    # =====================================================
    # LOAD STACK
    # =====================================================

    def load_stack(self):

        files = sorted(
            glob.glob(
                os.path.join(
                    self.inputs_dir,
                    'los', 
                    "tsf_los_*.tif"
                )
            )
        )
        # print(files) 

        self.files = files

        self.stack = []

        self.dates = []

        self.statistics = []

        for f in files:

            date = os.path.basename(
                f
            ).split("_")[-1].replace(
                ".tif",
                ""
            )

            with rasterio.open(f) as src:

                img = src.read(1)

            values = img[
                self.mask
            ]

            values = values[
                np.isfinite(values)
            ]

            self.statistics.append({

                "date": date,

                "mean":
                    float(
                        np.mean(values)
                    ),

                "std":
                    float(
                        np.std(values)
                    ),

                "min":
                    float(
                        np.min(values)
                    ),

                "max":
                    float(
                        np.max(values)
                    ),

                "median":
                    float(
                        np.median(values)
                    ),

                "mad":
                    float(
                        median_abs_deviation(
                            values
                        )
                    )
            })

            self.stack.append(img)

            self.dates.append(date)

        self.stack = np.array(
            self.stack
        )
        print(self.stack.shape) 

    # =====================================================
    # EXPORT STATISTICS
    # =====================================================

    def export_statistics(self):

        json_file = os.path.join(
            self.output_dir,
            "los_statistics.json"
        )

        with open(
            json_file,
            "w"
        ) as fp:

            json.dump(
                self.statistics,
                fp,
                indent=2
            )

        pd.DataFrame(
            self.statistics
        ).to_csv(

            os.path.join(
                self.output_dir,
                "temporal_statistics.csv"
            ),

            index=False
        )

    # =====================================================
    # HISTOGRAM
    # =====================================================

    def plot_histogram(self):

        latest = self.stack[-1]

        vals = latest[
            self.mask
        ]

        vals = vals[
            np.isfinite(vals)
        ]

        plt.figure(
            figsize=(8,5)
        )

        plt.hist(
            vals * 1000,
            bins=40
        )

        plt.xlabel(
            "LOS [mm]"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            "LOS Distribution Within TSF"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "los_histogram_latest.png"
            ),
            dpi=300
        )

        plt.close()

    # =====================================================
    # FIND REPRESENTATIVE PIXELS
    # =====================================================

    def select_pixels(self):

        final_img = self.stack[-1]

        masked = np.where(
            self.mask,
            final_img,
            np.nan
        )

        self.stable_y, self.stable_x = np.unravel_index(

            np.nanargmin(

                np.abs(
                    masked -
                    np.nanmedian(
                        masked
                    )
                )
            ),

            masked.shape
        )

        self.deformed_y, self.deformed_x = np.unravel_index(

            np.nanargmin(
                masked
            ),

            masked.shape
        )

        delta = (
            self.stack[-1]
            -
            self.stack[0]
        )

        delta[
            ~self.mask
        ] = np.nan

        self.anomaly_y, self.anomaly_x = np.unravel_index(

            np.nanargmin(
                delta
            ),

            delta.shape
        )

    # =====================================================
    # TEMPORAL PROFILES
    # =====================================================

    def plot_profiles(self):

        stable_ts = self.stack[
            :,
            self.stable_y,
            self.stable_x
        ]

        deformed_ts = self.stack[
            :,
            self.deformed_y,
            self.deformed_x
        ]

        anomaly_ts = self.stack[
            :,
            self.anomaly_y,
            self.anomaly_x
        ]

        plt.figure(
            figsize=(12,6)
        )

        plt.plot(
            stable_ts * 1000,
            label="Stable"
        )

        plt.plot(
            deformed_ts * 1000,
            label="Deformed"
        )

        plt.plot(
            anomaly_ts * 1000,
            label="Anomalous"
        )

        plt.xlabel(
            "Acquisition"
        )

        plt.ylabel(
            "LOS [mm]"
        )

        plt.legend()

        plt.grid()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "temporal_profiles.png"
            ),
            dpi=300
        )

        plt.close()

    # =====================================================
    # TEMPORAL STD MAP
    # =====================================================

    def temporal_std_map(self):

        std_map = np.nanstd(
            self.stack,
            axis=0
        )

        outfile = os.path.join(
            self.output_dir,
            "temporal_std_map.tif"
        )

        profile = self.profile.copy()

        profile.update(
            dtype="float32",
            count=1
        )

        with rasterio.open(
            outfile,
            "w",
            **profile
        ) as dst:

            dst.write(
                std_map.astype(
                    np.float32
                ),
                1
            )

    # =====================================================
    # TEMPORAL TREND MAP
    # =====================================================

    def temporal_trend_map(self):

        t = np.arange(
            self.stack.shape[0]
        )

        trend = np.full(
            self.stack.shape[1:],
            np.nan,
            dtype=np.float32
        )

        for row in range(
            trend.shape[0]
        ):

            for col in range(
                trend.shape[1]
            ):

                y = self.stack[
                    :,
                    row,
                    col
                ]

                valid = np.isfinite(
                    y
                )

                if np.sum(valid) < 5:

                    continue

                slope = np.polyfit(
                    t[valid],
                    y[valid],
                    1
                )[0]

                trend[
                    row,
                    col
                ] = slope

        outfile = os.path.join(
            self.output_dir,
            "temporal_trend_map.tif"
        )

        profile = self.profile.copy()

        profile.update(
            dtype="float32",
            count=1
        )

        with rasterio.open(
            outfile,
            "w",
            **profile
        ) as dst:

            dst.write(
                trend,
                1
            )

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        self.load_mask()

        self.load_stack()

        self.export_statistics()

        self.plot_histogram()

        self.select_pixels()

        self.plot_profiles()

        self.temporal_std_map()

        self.temporal_trend_map()

        print(
            "Slope EDA finished."
        )

### MAIN ### 
cfg_fn = 'slope_config.yaml'
with open(cfg_fn) as f:
    cfg = yaml.safe_load(f)

eda = SlopeEDA(cfg)
eda.run() 

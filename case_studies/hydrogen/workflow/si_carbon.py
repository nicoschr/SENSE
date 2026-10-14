"""Verify carbon conventions against the current archive without replacing it.

Load inventory matrices and recover activity levels from saved contributions.
This audit performs no inventory solve and changes no calculation archive.
"""

import io
import json
from zipfile import ZipFile

import bw2data as bd
import numpy as np
import pandas as pd
from bw2calc import LCA
from scipy import sparse

from sense.artifacts import write_json_atomic
from sense.utils.cache import file_hash

from . import case_config
from .carbon_methods import CLIMATE, EXPLICIT_FAMILY, SOURCE_FAMILY
from .si_common import require, write_csv


def run():
    cfg = case_config.load()
    archive_path = case_config.path("data_dir", cfg) / "paired_hydrogen/results.zip"
    out = case_config.path("data_dir", cfg) / "si/carbon"
    out.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(archive_path)
    require(
        cfg["brightway"]["project"] in bd.projects,
        "Study project is absent; build the backgrounds first",
    )
    bd.projects.set_current(cfg["brightway"]["project"])
    with ZipFile(archive_path) as z:

        def table(name):
            return pd.read_csv(
                z.open(name + ".csv"),
                keep_default_na=False,
                float_precision="round_trip",
                low_memory=False,
            )

        methods, activities, units = (
            table("methods"),
            table("activities"),
            table("functional_units"),
        )
        demand = {(row.activity_database, row.activity_code): 1e-12 for row in units.itertuples()}
        lca = LCA(demand)
        lca.load_lci_data()
        columns = [
            lca.activity_dict[key] for key in zip(activities.database, activities.code, strict=True)
        ]
        bio = lca.biosphere_matrix[:, columns]

        def coefficient(key):
            cf = np.zeros(bio.shape[0])
            for flow, value in bd.Method(key).load():
                index = lca.biosphere_dict.get(tuple(flow))
                if index is not None:
                    cf[index] = float(np.float32(value))
            return np.asarray(cf @ bio).ravel()

        reference = methods[methods.kind.isin(["environmental", "working_hours"])]
        direct = np.vstack(
            [coefficient(tuple(json.loads(row.method))) for row in reference.itertuples()]
        )
        selector = np.argmax(np.abs(direct), axis=0)
        indices = np.arange(len(activities))
        selected_coefficient = direct[selector, indices]
        active = selected_coefficient != 0
        standard_key = (SOURCE_FAMILY + " regionalized", *CLIMATE)
        explicit_key = (EXPLICIT_FAMILY + " regionalized", *CLIMATE)
        conventional, explicit = coefficient(standard_key), coefficient(explicit_key)
        if np.any(conventional[~active] != 0) or np.any(explicit[~active] != 0):
            raise ValueError("Unrecoverable activity levels have nonzero climate coefficients")
        maximum_error, results = 0.0, []
        for unit in units.itertuples():
            archived = sparse.load_npz(io.BytesIO(z.read(unit.contribution_file))).tocsr()
            actual = archived[reference.method_index.to_numpy()].toarray()
            production = np.zeros(len(activities))
            production[active] = (
                actual[selector[active], indices[active]] / selected_coefficient[active]
            )
            reproduced = direct * production
            error = float(np.max(np.abs(reproduced - actual)))
            if not np.allclose(reproduced, actual, rtol=1e-9, atol=1e-10):
                raise ValueError(f"Contribution reconstruction failed: {error}")
            maximum_error = max(maximum_error, error)
            results.append(
                {
                    "technology": unit.technology,
                    "country": unit.country,
                    "scenario": unit.scenario,
                    "conventional_climate_score": float(conventional @ production),
                    "explicit_climate_score": float(explicit @ production),
                }
            )
            print(f"Verified saved inventory: {unit.fu_index + 1}/{len(units)}", flush=True)
    if file_hash(archive_path) != source_hash:
        raise ValueError("Calculation archive changed during verification")
    current = pd.DataFrame(results)
    write_csv(current, out / "carbon_convention_comparison.csv")
    evidence = {
        "basis": "Current archived activity levels, reconstructed without an inventory solve",
        "reference_methods": len(reference),
        "product_systems": len(results),
        "maximum_absolute_reconstruction_difference": maximum_error,
        "relative_tolerance": 1e-9,
        "absolute_tolerance": 1e-10,
        "conventional_method": standard_key,
        "explicit_method": explicit_key,
        "source_archive_sha256": source_hash,
        "archive_modified": False,
    }
    write_json_atomic(out / "evidence.json", evidence)
    print("Compared carbon conventions for the current archive. Archive unchanged.")

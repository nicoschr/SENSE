"""Backgrounds stage and calculation helpers."""

from collections import Counter, defaultdict

import bw2data as bd
import bw2io as bi
import pandas as pd
import regioinvent

from .case_config import (
    TRADE_DATABASE_PATH,
    ecoinvent_database_name,
    load,
    require_ecoinvent_credentials,
)
from .study_checks import check_backgrounds


def patch_methods_and_print_replaced_biospheres(
    *,
    standard_biosphere_name: str,
    allowed_flow_dbs: set[str],
    dry_run: bool = True,
    validate_standard_flow_exists: bool = True,
    patch_only_if_db_name_contains: tuple[str, ...] | None = ("biosphere",),
    method_family_contains: str | None = None,
    max_methods_print: int = 40,
) -> pd.DataFrame:
    """Remap eligible method flow references and return an audit DataFrame.

    Scan methods in the active project, optionally limited by ``method_family_contains``.
    Retain references in ``allowed_flow_dbs``; apply ``patch_only_if_db_name_contains``
    to other database names. Preserve factors and additional row fields when replacing
    references with ``standard_biosphere_name``.

    With the default existence check, skip codes absent from the standard database.
    Disabling that check permits unchecked references. ``dry_run=True`` reports
    proposals only; otherwise write changed methods. Raise ``ValueError`` if the
    standard database is missing. The audit reports rewritten and skipped row counts.
    """
    if standard_biosphere_name not in bd.databases:
        raise ValueError(
            f"Standard biosphere '{standard_biosphere_name}' not found in this project."
        )

    allowed_flow_dbs = set(allowed_flow_dbs)
    allowed_flow_dbs.add(standard_biosphere_name)

    standard_db = bd.Database(standard_biosphere_name)

    def standard_flow_exists(code: str) -> bool:
        if not validate_standard_flow_exists:
            return True
        try:
            standard_db.get(code)
            return True
        except Exception:
            return False

    scanned = 0
    candidate_methods = 0
    methods_with_rewrites = 0

    # diagnostics containers
    per_method_replaced: dict[str, Counter] = defaultdict(
        Counter
    )  # method -> Counter(src_db -> count)
    global_replaced: Counter = Counter()  # src_db -> total rows replaced
    per_method_skipped_missing: Counter = Counter()  # method -> skipped rows

    rows = []

    for m in list(bd.methods):
        if method_family_contains and method_family_contains.lower() not in m[0].lower():
            continue

        data = bd.Method(m).load()
        if not data:
            continue

        scanned += 1
        referenced = {row[0][0] for row in data}
        disallowed = referenced - allowed_flow_dbs

        if patch_only_if_db_name_contains is not None:
            disallowed = {
                dbn
                for dbn in disallowed
                if any(k.lower() in dbn.lower() for k in patch_only_if_db_name_contains)
            }

        if not disallowed:
            continue

        candidate_methods += 1

        new_data = []
        rewritten = 0
        skipped = 0

        for row in data:
            fk, cf, *rest = row
            dbn, code = fk

            if dbn in disallowed:
                if standard_flow_exists(code):
                    fk = (standard_biosphere_name, code)
                    rewritten += 1
                    per_method_replaced[str(m)][dbn] += 1
                    global_replaced[dbn] += 1
                else:
                    skipped += 1

            new_data.append((fk, cf, *rest))

        if rewritten > 0:
            methods_with_rewrites += 1
            if not dry_run:
                bd.Method(m).write(new_data)

        per_method_skipped_missing[str(m)] += skipped

        rows.append(
            {
                "method": str(m),
                "rows_total": len(data),
                "disallowed_db_refs": "; ".join(sorted(disallowed)),
                "rows_rewritten": rewritten,
                "rows_skipped_missing_standard_flow": skipped,
                "replaced_from": "; ".join(
                    f"{src} ({cnt})" for src, cnt in per_method_replaced[str(m)].most_common()
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No candidate methods found (no disallowed flow DB refs under current settings).")
        return df
    df = df.sort_values(["rows_rewritten", "rows_total"], ascending=False)

    # ---- PRINT DIAGNOSTICS ----
    print(f"Standard biosphere: {standard_biosphere_name}")
    print(f"Allowed flow DBs: {sorted(allowed_flow_dbs)}")
    print(f"Dry run: {dry_run}")
    print(f"Scanned methods: {scanned}")
    print(f"Candidate methods (had disallowed refs): {candidate_methods}")
    print(f"Methods with rewrites: {methods_with_rewrites}")
    print()

    if global_replaced:
        print("Biosphere DBs replaced (global totals):")
        for src, cnt in global_replaced.most_common():
            print(f"  {src}  ->  {standard_biosphere_name}   ({cnt} CF rows)")
    else:
        print("No replacements performed / possible under current settings.")
        print(
            "If you expected replacements, the most common reason is UUIDs missing in the standard biosphere"
        )
        print("while validate_standard_flow_exists=True.")
    print()

    if not df.empty:
        to_show = df[df["rows_rewritten"] > 0].head(max_methods_print)
        if to_show.empty:
            # show top candidates anyway
            to_show = df.head(max_methods_print)
            print(f"Top {len(to_show)} candidate methods (no rewrites):")
        else:
            print(f"Top {len(to_show)} methods with rewrites:")

        print(
            to_show[
                [
                    "method",
                    "rows_rewritten",
                    "rows_skipped_missing_standard_flow",
                    "replaced_from",
                    "disallowed_db_refs",
                ]
            ].to_string(index=False)
        )
    else:
        print("No candidate methods found.")

    return df


def run(*, build=False):
    """Execute this stage using the configured study inputs."""
    CFG = load()
    PROJECT = CFG["brightway"]["project"]
    EI_VERSION = CFG["versions"]["ecoinvent"]
    EI_NAME = ecoinvent_database_name(CFG)
    REBUILD_BACKGROUNDS = build
    EI_REGIONALIZED = ecoinvent_database_name(CFG, regionalized=True)
    REGIO_DB = CFG["brightway"]["regioinvent_database"]

    if not REBUILD_BACKGROUNDS and PROJECT not in bd.projects:
        raise ValueError("Project absent; run build-backgrounds for initial construction")

    bd.projects.set_current(PROJECT)

    if build and (EI_REGIONALIZED in bd.databases or REGIO_DB in bd.databases):
        raise ValueError(
            "Regionalized backgrounds already exist; use check-backgrounds. "
            "Choose a new project for an independent build."
        )
    REUSE_BACKGROUNDS = not REBUILD_BACKGROUNDS

    if REUSE_BACKGROUNDS:
        print(check_backgrounds(EI_NAME, EI_REGIONALIZED, REGIO_DB))

    if not REUSE_BACKGROUNDS:
        # 1) Ensure project exists / is selected
        bd.projects.set_current(PROJECT)

        # 2) Import the configured ecoinvent cutoff database if missing
        if EI_NAME in bd.databases:
            print(f"Database '{EI_NAME}' already exists in project '{bd.projects.current}'.")
        else:
            username, password = require_ecoinvent_credentials(CFG)
            bi.import_ecoinvent_release(
                version=EI_VERSION,
                system_model="cutoff",
                username=username,
                password=password,
                use_mp=False,
            )
            print(f"Database '{EI_NAME}' installed successfully.")
    BW_PROJECT = PROJECT
    EI_DB_NAME = EI_NAME
    TRADE_DB_PATH = TRADE_DATABASE_PATH
    OUTPUT_DB_NAME = CFG["brightway"]["regioinvent_database"]
    TRADE_CUTOFF = 0.99  # 0 to 1; higher values take longer (2-3 hours)

    bd.projects.set_current(BW_PROJECT)

    print(bd.databases)

    if not REUSE_BACKGROUNDS:
        # 1. Initialize the Regioinvent class
        print(f"Initializing Regioinvent for project: {BW_PROJECT}...")
        regio = regioinvent.Regioinvent(
            bw_project_name=BW_PROJECT,
            ecoinvent_database_name=EI_DB_NAME,
            ecoinvent_version=EI_VERSION,
        )

    if not REUSE_BACKGROUNDS:
        # 2. Spatialize the elementary flows
        # Creates 'biosphere3_spatialized_flows' and a regionalized copy of ecoinvent
        print("Step 1: Spatializing ecoinvent database...")
        regio.spatialize_my_ecoinvent()

    if not REUSE_BACKGROUNDS:
        # 3. Import LCIA methods compatible with spatialized flows
        print("Step 3: Importing regionalized LCIA methods...")
        regio.import_fully_regionalized_impact_method(lcia_method="all")

        print("Step 3b: Harmonizing biosphere references inside regionalized methods...")
        allowed = {
            "biosphere3_spatialized_flows",
            "sociosphere",
            f"ecoinvent-{EI_VERSION}-biosphere",
        }

        patch_methods_and_print_replaced_biospheres(
            standard_biosphere_name=f"ecoinvent-{EI_VERSION}-biosphere",
            allowed_flow_dbs=allowed,
            dry_run=False,  # Apply the configured method-reference changes
            max_methods_print=5,
        )

    if not REUSE_BACKGROUNDS:
        # 4. Regionalize with trade data
        # This step is the long one (2-3 hours depending on cutoff)
        print(f"Step 3: Regionalizing with trade data (Cutoff: {TRADE_CUTOFF})...")
        regio.regionalize_ecoinvent_with_trade(
            trade_database_path=TRADE_DB_PATH,
            target_database_name=OUTPUT_DB_NAME,
            cutoff=TRADE_CUTOFF,
        )

    check_backgrounds(EI_NAME, EI_REGIONALIZED, REGIO_DB)

"""Unified Data Hub: manual-download AND synthetic loading for every
PRAMANA dataset family.

PRAMANA ships with NO built-in observational data. Each of the five
families (SN Ia, BAO, CMB, JWST-era, Weak Lensing) is loaded through one of three paths:
  - 📥 Download & Upload : official release links + file upload
  - 📂 File Path         : point at files already on disk
  - 🧪 Synthetic         : generate realistic mock data for testing

All loaded data is stored in st.session_state[key] so downstream pages
(fit, tension, forecast, compression, ...) can reuse it.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from pramana.web.components.data_loader import _load_pantheon_cached
from pramana.core.synthetic import (
    synthetic_desi_bao,
    synthetic_highz_sn,
    synthetic_act_cmb,
    synthetic_h0_tables,
    synthetic_s8_tables,
)
from pramana.core.jwst_probes import H0_MEASUREMENTS, S8_MEASUREMENTS

# ---------------------------------------------------------------------
# Dataset catalog — the single source of truth for what the hub offers
# ---------------------------------------------------------------------

DATASET_CATALOG = {
    "pantheon": {
        "title": "SN Ia — Pantheon+SH0ES",
        "short": "SN",
        "description": (
            "Type Ia supernova Hubble diagram with full systematic "
            "covariance (shape-only, H0-marginalized)."
        ),
        "sources": [
            {
                "label": "Pantheon+SH0ES Data Release (GitHub)",
                "url": "https://github.com/PantheonPlusSH0ES/DataRelease",
            }
        ],
        "files": (
            "`Pantheon+SH0ES.dat` (whitespace; zHD, m_b_corr, IS_CALIBRATOR) + "
            "`Pantheon+SH0ES_STAT+SYS.cov` (first line N, then N×N values)"
        ),
        "default_key": "pantheon_data",
    },
    "bao": {
        "title": "BAO — DESI DR2",
        "short": "BAO",
        "description": (
            "Baryon Acoustic Oscillation distance-ratio measurements "
            "(DM/rd, DH/rd, DV/rd) from DESI DR2."
        ),
        "sources": [
            {
                "label": "DESI DR2 Data Release",
                "url": "https://data.desi.lbl.gov/doc/releases/dr2/",
            },
            {
                "label": "DESI DR2 BAO paper (arXiv)",
                "url": "https://arxiv.org/abs/2504.18498",
            },
        ],
        "files": (
            "CSV with columns: `tracer, z, DM_rd, DM_rd_err, DH_rd, DH_rd_err, "
            "rho_MH, DV_rd, DV_rd_err` (use the downloadable template)"
        ),
        "default_key": "bao_data",
    },
    "cmb": {
        "title": "CMB — ACT DR6",
        "short": "CMB",
        "description": (
            "CMB lensing convergence (C_kk) + primary spectra (TT/EE/TE/BB) "
            "in C_ell, uK^2 convention."
        ),
        "sources": [
            {
                "label": "ACT DR6 likelihood data (LAMBDA)",
                "url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/",
            }
        ],
        "files": (
            "Numpy `.npz` archive with arrays: `ell, cl_tt, cl_ee, cl_te, "
            "cl_bb, ell_kk, cl_kk`"
        ),
        "default_key": "cmb_data",
    },
    "jwst": {
        "title": "JWST-era Probes",
        "short": "JWST",
        "description": (
            "High-z SNe extension, H0 tension compilation, and S8 growth "
            "tension compilation."
        ),
        "sources": [
            {
                "label": "H0 measurements (SH0ES / CCHP references)",
                "url": "https://ui.adsabs.harvard.edu/abs/2022ApJ...934..186R/abstract",
            },
            {
                "label": "S8 measurements (DES / KiDS / HSC)",
                "url": "https://ui.adsabs.harvard.edu/abs/2021PhRvD.104f3507H/abstract",
            },
        ],
        "files": (
            "Three optional CSV uploads (see Download tab for templates): "
            "H0 table, S8 table, high-z SN list."
        ),
        "default_key": "jwst_data",
    },
    "weak_lensing": {
        "title": "Weak Lensing — Euclid & Rubin/LSST (Cosmic Shear)",
        "short": "WL",
        "description": (
            "Tomographic cosmic shear 2-point functions (shear-shear) from "
            "Euclid and Rubin Observatory LSST. Forward-model infrastructure "
            "validated on synthetic data only — no public cosmology-ready "
            "shear catalogs exist yet (Euclid first results ~2027, LSST 10-yr "
            "survey started 2026)."
        ),
        "sources": [
            {
                "label": "Euclid Consortium data access",
                "url": "https://www.euclid-ec.org/",
            },
            {
                "label": "Rubin/LSST data access (RSP)",
                "url": "https://www.lsst.org/",
            },
        ],
        "files": (
            "Numpy `.npz` archive with arrays: `data_vector, cov, pairs, "
            "ell_array, truth` — or CSV long-form with columns "
            "`pair_i, pair_j, ell, C_ell, sigma`"
        ),
        "default_key": "shear_data",
    },
}

DOWNLOAD_URLS = {
    "pantheon": "https://github.com/PantheonPlusSH0ES/DataRelease",
    "bao": "https://data.desi.lbl.gov/doc/releases/dr2/",
    "cmb": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/",
    "weak_lensing": "https://www.euclid-ec.org/",
}


# ---------------------------------------------------------------------
# BAO helpers
# ---------------------------------------------------------------------

BAO_TEMPLATE_COLUMNS = [
    "tracer", "z", "DM_rd", "DM_rd_err", "DH_rd", "DH_rd_err", "rho_MH", "DV_rd", "DV_rd_err",
]


def _bao_to_rows() -> list[dict]:
    from pramana.core.bao_desi import DESI_DR2_BAO_TABLE

    rows = []
    for tracer, d in DESI_DR2_BAO_TABLE.items():
        rows.append({
            "tracer": tracer,
            "z": d["z"],
            "DM_rd": d.get("DM_rd", ""),
            "DM_rd_err": d.get("DM_rd_err", ""),
            "DH_rd": d.get("DH_rd", ""),
            "DH_rd_err": d.get("DH_rd_err", ""),
            "rho_MH": d.get("rho_MH", ""),
            "DV_rd": d.get("DV_rd", ""),
            "DV_rd_err": d.get("DV_rd_err", ""),
        })
    return rows


def _rows_to_bao(rows: list[dict]):
    """Rebuild the DESI data vector + block-diagonal covariance from rows."""
    labels, z_list, data, blocks = [], [], [], []
    for r in rows:
        tracer = str(r["tracer"])
        z = float(r["z"])
        if str(r["DV_rd"]).strip() != "" and r["DV_rd"] is not None:
            labels.append((tracer, "DV_rd"))
            z_list.append(z)
            data.append(float(r["DV_rd"]))
            blocks.append(np.array([[float(r["DV_rd_err"]) ** 2]]))
        else:
            labels.append((tracer, "DM_rd"))
            labels.append((tracer, "DH_rd"))
            z_list.extend([z, z])
            data.extend([float(r["DM_rd"]), float(r["DH_rd"])])
            sM, sH = float(r["DM_rd_err"]), float(r["DH_rd_err"])
            rho = float(r["rho_MH"]) if str(r["rho_MH"]).strip() != "" else 0.0
            blocks.append(np.array([[sM**2, rho * sM * sH], [rho * sM * sH, sH**2]]))

    data = np.array(data)
    n = len(data)
    cov = np.zeros((n, n))
    i = 0
    for block in blocks:
        k = block.shape[0]
        cov[i : i + k, i : i + k] = block
        i += k
    return labels, np.array(z_list), data, cov


# ---------------------------------------------------------------------
# Per-family loaders — each returns the session-state dict
# ---------------------------------------------------------------------

def _load_pantheon_upload(dat_bytes, cov_bytes) -> dict:
    z, mb, cov, df = _load_pantheon_cached(dat_bytes, cov_bytes, None, None)
    return {"z": z, "mb_obs": mb, "cov": cov, "df": df, "source": "upload"}


def _load_pantheon_path(data_path, cov_path) -> dict:
    z, mb, cov, df = _load_pantheon_cached(None, None, data_path, cov_path)
    return {"z": z, "mb_obs": mb, "cov": cov, "df": df, "source": "path"}


def _load_bao_upload(uploaded) -> dict:
    df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=r"[,\s]+")
    rows = df.to_dict("records")
    labels, z_arr, data, cov = _rows_to_bao(rows)
    return {"labels": labels, "z_arr": z_arr, "data": data, "cov": cov, "source": "upload"}


def _load_bao_path(path) -> dict:
    df = pd.read_csv(path, sep=r"[,\s]+")
    rows = df.to_dict("records")
    labels, z_arr, data, cov = _rows_to_bao(rows)
    return {"labels": labels, "z_arr": z_arr, "data": data, "cov": cov, "source": "path"}


def _load_cmb_upload(uploaded) -> dict:
    with np.load(io.BytesIO(uploaded.getvalue())) as npz:
        return {
            "ell": npz["ell"], "cl_tt": npz["cl_tt"], "cl_ee": npz["cl_ee"],
            "cl_te": npz["cl_te"], "cl_bb": npz["cl_bb"],
            "ell_kk": npz["ell_kk"], "cl_kk": npz["cl_kk"],
            "source": "upload",
        }


def _load_cmb_path(path) -> dict:
    with np.load(path) as npz:
        return {
            "ell": npz["ell"], "cl_tt": npz["cl_tt"], "cl_ee": npz["cl_ee"],
            "cl_te": npz["cl_te"], "cl_bb": npz["cl_bb"],
            "ell_kk": npz["ell_kk"], "cl_kk": npz["cl_kk"],
            "source": "path",
        }


def _load_jwst_upload(h0_file, s8_file, sn_file) -> dict:
    out = {"source": "upload"}
    if h0_file is not None:
        df = pd.read_csv(io.BytesIO(h0_file.getvalue()))
        out["h0_table"] = {
            row["name"]: {"H0": float(row["H0"]), "err": float(row["err"]),
                           "family": str(row["family"])}
            for row in df.to_dict("records")
        }
    if s8_file is not None:
        df = pd.read_csv(io.BytesIO(s8_file.getvalue()))
        out["s8_table"] = {
            row["name"]: {"S8": float(row["S8"]), "err": float(row["err"]),
                           "family": str(row["family"])}
            for row in df.to_dict("records")
        }
    if sn_file is not None:
        df = pd.read_csv(io.BytesIO(sn_file.getvalue()), sep=r"[,\s]+")
        out["highz_sn"] = {
            "z": df["z"].values, "mb": df["mb"].values, "err": df["err"].values
        }
    return out


# ---------------------------------------------------------------------
# Status / summary helpers
# ---------------------------------------------------------------------

def _family_status(key: str) -> str:
    if key not in st.session_state:
        return "❌ Not loaded"
    src = st.session_state[key].get("source", "unknown")
    return "✅ Loaded" + (f" (synthetic)" if src == "synthetic" else " (real)")


def _status_html(short: str, key: str) -> str:
    txt = _family_status(key)
    color = "#2f9e44" if "✅" in txt else "#a83232"
    return (
        f"<span style='color:{color};font-family:JetBrains Mono,monospace'>"
        f"{txt.replace('❌', '○').replace('✅', '●')}</span>"
    )


# ---------------------------------------------------------------------
# The per-family tabbed loader
# ---------------------------------------------------------------------

def _render_download_tab(family_id: str, key: str):
    spec = DATASET_CATALOG[family_id]
    st.markdown(f"**Official sources**")
    for s in spec["sources"]:
        st.markdown(f"- [{s['label']}]({s['url']})")
    st.markdown(f"**Expected files:** {spec['files']}")

    if family_id == "pantheon":
        col1, col2 = st.columns(2)
        with col1:
            dat_file = st.file_uploader("Pantheon+ data (.dat)", type=["dat", "txt"], key=f"{key}_hub_dat")
        with col2:
            cov_file = st.file_uploader("Covariance (.cov)", type=["cov", "txt"], key=f"{key}_hub_cov")
        if st.button("Load Uploaded Files", type="primary", key=f"{key}_hub_load"):
            if dat_file is None or cov_file is None:
                st.error("Please upload both .dat and .cov files")
            else:
                try:
                    st.session_state[key] = _load_pantheon_upload(dat_file.read(), cov_file.read())
                    st.success(f"Loaded {len(st.session_state[key]['z'])} SNe from upload")
                except Exception as e:
                    st.error(f"Error loading files: {e}")

    elif family_id == "bao":
        st.download_button(
            "Download DESI DR2 reference template (.csv)",
            data=pd.DataFrame(_bao_to_rows()).to_csv(index=False).encode(),
            file_name="desi_dr2_bao_template.csv",
            mime="text/csv",
            key=f"{key}_hub_bao_tpl",
        )
        bao_file = st.file_uploader("BAO CSV", type=["csv", "txt"], key=f"{key}_hub_bao")
        if st.button("Load BAO File", type="primary", key=f"{key}_hub_load"):
            if bao_file is None:
                st.error("Please upload a BAO CSV file")
            else:
                try:
                    st.session_state[key] = _load_bao_upload(bao_file)
                    st.success(f"Loaded {len(st.session_state[key]['data'])} BAO data points")
                except Exception as e:
                    st.error(f"Error loading BAO file: {e}")

    elif family_id == "cmb":
        st.markdown(
            "Download the ACT DR6 likelihood data from [LAMBDA]"
            f"({spec['sources'][0]['url']}). For the Data Hub, save the "
            "spectra you want to use as a `.npz` (see Expected files)."
        )
        cmb_file = st.file_uploader("CMB spectra (.npz)", type=["npz"], key=f"{key}_hub_cmb")
        if st.button("Load CMB Spectra", type="primary", key=f"{key}_hub_load"):
            if cmb_file is None:
                st.error("Please upload a .npz archive")
            else:
                try:
                    st.session_state[key] = _load_cmb_upload(cmb_file)
                    st.success("Loaded synthetic/full CMB spectra from upload")
                except Exception as e:
                    st.error(f"Error loading CMB file: {e}")

    elif family_id == "jwst":
        st.markdown("**H0 table template**")
        h0_tpl = pd.DataFrame([
            {"name": k, "H0": v["H0"], "err": v["err"], "family": v["family"]}
            for k, v in H0_MEASUREMENTS.items()
        ])
        st.download_button(
            "Download H0 reference template (.csv)",
            data=h0_tpl.to_csv(index=False).encode(),
            file_name="h0_template.csv", mime="text/csv",
            key=f"{key}_hub_h0_tpl",
        )
        st.markdown("**S8 table template**")
        s8_tpl = pd.DataFrame([
            {"name": k, "S8": v["S8"], "err": v["err"], "family": v["family"]}
            for k, v in S8_MEASUREMENTS.items()
        ])
        st.download_button(
            "Download S8 reference template (.csv)",
            data=s8_tpl.to_csv(index=False).encode(),
            file_name="s8_template.csv", mime="text/csv",
            key=f"{key}_hub_s8_tpl",
        )
        st.markdown("**High-z SNe template** — CSV with columns `z, mb, err`.")

        col1, col2, col3 = st.columns(3)
        with col1:
            h0_file = st.file_uploader("H0 table (.csv)", type=["csv"], key=f"{key}_hub_h0")
        with col2:
            s8_file = st.file_uploader("S8 table (.csv)", type=["csv"], key=f"{key}_hub_s8")
        with col3:
            sn_file = st.file_uploader("High-z SNe (.csv)", type=["csv"], key=f"{key}_hub_sn")
        if st.button("Load Uploaded Files", type="primary", key=f"{key}_hub_load"):
            try:
                st.session_state[key] = _load_jwst_upload(h0_file, s8_file, sn_file)
                st.success("Loaded JWST-era probes from upload")
            except Exception as e:
                st.error(f"Error loading JWST files: {e}")

    elif family_id == "weak_lensing":
        st.markdown("""
**No public shear catalogs available yet.** Euclid's first cosmology results
are expected ~2027; Rubin/LSST's 10-year survey started June 2026.

Use the **Synthetic** tab to generate mock shear data vectors for testing
the full pipeline. When real data becomes available, it should be provided
as a `.npz` with arrays: `data_vector, cov, pairs, ell_array` or as a
CSV with columns `pair_i, pair_j, ell, C_ell, sigma`.
""")
        shear_file = st.file_uploader("Shear data (.npz or .csv)", type=["npz", "csv"], key=f"{key}_hub_shear")
        if st.button("Load Shear Data", type="primary", key=f"{key}_hub_load"):
            if shear_file is None:
                st.error("Please upload a .npz or .csv file")
            else:
                try:
                    if shear_file.name.endswith(".npz"):
                        with np.load(io.BytesIO(shear_file.getvalue())) as npz:
                            st.session_state[key] = {
                                "data_vector": npz["data_vector"], "cov": npz["cov"],
                                "pairs": npz["pairs"], "ell_array": npz["ell_array"],
                                "source": "upload", "preset": npz.get("preset", "unknown"),
                            }
                    else:
                        df = pd.read_csv(io.BytesIO(shear_file.getvalue()))
                        # Expect long-form: pair_i, pair_j, ell, C_ell, sigma
                        # For simplicity, just store the dataframe
                        st.session_state[key] = {"df": df, "source": "upload", "preset": "custom"}
                    st.success("Loaded shear data from upload")
                except Exception as e:
                    st.error(f"Error loading shear file: {e}")


def _render_path_tab(family_id: str, key: str):
    spec = DATASET_CATALOG[family_id]
    defaults = {
        "pantheon": ("data/pantheon/Pantheon+SH0ES.dat", "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"),
        "bao": ("data/desi/desi_dr2_bao.csv",),
        "cmb": ("data/act/act_dr6_spectra.npz",),
        "jwst": ("data/jwst/h0_table.csv", "data/jwst/s8_table.csv", "data/jwst/highz_sn.csv"),
    }[family_id]

    if family_id == "pantheon":
        col1, col2 = st.columns(2)
        with col1:
            dp = st.text_input("Data file (.dat)", defaults[0], key=f"{key}_hub_dat_path")
        with col2:
            cp = st.text_input("Covariance file (.cov)", defaults[1], key=f"{key}_hub_cov_path")
        if st.button("Load from Path", type="primary", key=f"{key}_hub_path_load"):
            if not Path(dp).exists() or not Path(cp).exists():
                st.error("One or both files not found. Use Upload or Synthetic instead.")
            else:
                try:
                    st.session_state[key] = _load_pantheon_path(dp, cp)
                    st.success(f"Loaded {len(st.session_state[key]['z'])} SNe from path")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif family_id == "bao":
        bp = st.text_input("BAO CSV path", defaults[0], key=f"{key}_hub_bao_path")
        if st.button("Load from Path", type="primary", key=f"{key}_hub_path_load"):
            if not Path(bp).exists():
                st.error("File not found. Use Upload or Synthetic instead.")
            else:
                try:
                    st.session_state[key] = _load_bao_path(bp)
                    st.success(f"Loaded {len(st.session_state[key]['data'])} BAO data points")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif family_id == "cmb":
        cp = st.text_input("CMB .npz path", defaults[0], key=f"{key}_hub_cmb_path")
        if st.button("Load from Path", type="primary", key=f"{key}_hub_path_load"):
            if not Path(cp).exists():
                st.error("File not found. Use Upload or Synthetic instead.")
            else:
                try:
                    st.session_state[key] = _load_cmb_path(cp)
                    st.success("Loaded CMB spectra from path")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif family_id == "jwst":
        h0p = st.text_input("H0 table path", defaults[0], key=f"{key}_hub_h0_path")
        s8p = st.text_input("S8 table path", defaults[1], key=f"{key}_hub_s8_path")
        snp = st.text_input("High-z SNe path (optional)", defaults[2], key=f"{key}_hub_sn_path")
        if st.button("Load from Path", type="primary", key=f"{key}_hub_path_load"):
            try:
                h0f = Path(h0p) if h0p and Path(h0p).exists() else None
                s8f = Path(s8p) if s8p and Path(s8p).exists() else None
                snf = Path(snp) if snp and Path(snp).exists() else None
                if h0f is None and s8f is None:
                    st.error("Provide at least one existing H0 or S8 path")
                else:
                    import pandas as _pd
                    out = {"source": "path"}
                    if h0f:
                        _df = _pd.read_csv(h0f)
                        out["h0_table"] = {r["name"]: {"H0": float(r["H0"]), "err": float(r["err"]), "family": str(r["family"])} for r in _df.to_dict("records")}
                    if s8f:
                        _df = _pd.read_csv(s8f)
                        out["s8_table"] = {r["name"]: {"S8": float(r["S8"]), "err": float(r["err"]), "family": str(r["family"])} for r in _df.to_dict("records")}
                    if snf:
                        _df = _pd.read_csv(snf, sep=r"[,\s]+")
                        out["highz_sn"] = {"z": _df["z"].values, "mb": _df["mb"].values, "err": _df["err"].values}
                    st.session_state[key] = out
                    st.success("Loaded JWST-era probes from path")
            except Exception as e:
                st.error(f"Error: {e}")

    elif family_id == "weak_lensing":
        sp = st.text_input("Shear data path (.npz or .csv)", "data/wl/shear_data.npz", key=f"{key}_hub_shear_path")
        if st.button("Load from Path", type="primary", key=f"{key}_hub_path_load"):
            if not Path(sp).exists():
                st.error("File not found. Use Upload or Synthetic instead.")
            else:
                try:
                    if sp.endswith(".npz"):
                        with np.load(sp) as npz:
                            st.session_state[key] = {
                                "data_vector": npz["data_vector"], "cov": npz["cov"],
                                "pairs": npz["pairs"], "ell_array": npz["ell_array"],
                                "source": "path", "preset": npz.get("preset", "unknown"),
                            }
                    else:
                        df = pd.read_csv(sp)
                        st.session_state[key] = {"df": df, "source": "path", "preset": "custom"}
                    st.success("Loaded shear data from path")
                except Exception as e:
                    st.error(f"Error: {e}")


def _render_synthetic_tab(family_id: str, key: str):
    spec = DATASET_CATALOG[family_id]

    if family_id == "pantheon":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_syn = st.number_input("Number of SNe", 50, 2000, 300, key=f"{key}_hub_n")
        with col2:
            om_syn = st.number_input("True Ωₘ", 0.05, 0.6, 0.3, key=f"{key}_hub_om")
        with col3:
            seed = st.number_input("Seed", 0, 9999, 42, key=f"{key}_hub_seed")
        if st.button("Generate Synthetic", type="primary", key=f"{key}_hub_gen"):
            from pramana.core.data_io import make_synthetic_dataset
            z, mb, cov = make_synthetic_dataset(n=n_syn, seed=int(seed), Om_true=om_syn)
            st.session_state[key] = {"z": z, "mb_obs": mb, "cov": cov, "df": None, "source": "synthetic"}
            st.success(f"Generated {len(z)} synthetic SNe")

    elif family_id == "bao":
        col1, col2, col3 = st.columns(3)
        with col1:
            om_syn = st.number_input("True Ωₘ", 0.05, 0.6, 0.3, key=f"{key}_hub_om")
        with col2:
            h0_syn = st.number_input("H₀ (km/s/Mpc)", 50.0, 90.0, 70.0, key=f"{key}_hub_h0")
        with col3:
            seed = st.number_input("Seed", 0, 9999, 42, key=f"{key}_hub_seed")
        if st.button("Generate Synthetic", type="primary", key=f"{key}_hub_gen"):
            labels, z_arr, data, cov = synthetic_desi_bao(seed=int(seed), Om=om_syn, H0=h0_syn)
            st.session_state[key] = {"labels": labels, "z_arr": z_arr, "data": data, "cov": cov, "source": "synthetic"}
            st.success(f"Generated {len(data)} synthetic BAO data points")

    elif family_id == "cmb":
        col1, col2 = st.columns(2)
        with col1:
            lmax = st.number_input("ℓ_max", 500, 5000, 3000, key=f"{key}_hub_lmax")
        with col2:
            seed = st.number_input("Seed", 0, 9999, 42, key=f"{key}_hub_seed")
        if st.button("Generate Synthetic", type="primary", key=f"{key}_hub_gen"):
            st.session_state[key] = synthetic_act_cmb(seed=int(seed), lmax=int(lmax))
            st.session_state[key]["source"] = "synthetic"
            st.success("Generated synthetic ACT DR6-like spectra")

    elif family_id == "jwst":
        col1, col2 = st.columns(2)
        with col1:
            h0_true = st.number_input("True H₀", 60.0, 80.0, 70.0, key=f"{key}_hub_h0_true")
            s8_true = st.number_input("True S₈", 0.6, 1.0, 0.8, key=f"{key}_hub_s8_true")
        with col2:
            seed = st.number_input("Seed", 0, 9999, 42, key=f"{key}_hub_seed")
            n_hz = st.number_input("High-z SNe count", 5, 200, 40, key=f"{key}_hub_n_hz")
        if st.button("Generate Synthetic", type="primary", key=f"{key}_hub_gen"):
            z_sn, mb_sn, err_sn = synthetic_highz_sn(n=int(n_hz), seed=int(seed))
            st.session_state[key] = {
                "h0_table": synthetic_h0_tables(seed=int(seed), h0_true=h0_true),
                "s8_table": synthetic_s8_tables(seed=int(seed), s8_true=s8_true),
                "highz_sn": {"z": z_sn, "mb": mb_sn, "err": err_sn},
                "source": "synthetic",
            }
            st.success("Generated synthetic JWST-era probes")

    elif family_id == "weak_lensing":
        st.markdown("**Requires CAMB** — synthetic shear generation needs the CAMB "
                    "Boltzmann solver for nonlinear P(k,z). Install with `uv add camb`.")
        
        col1, col2 = st.columns(2)
        with col1:
            preset = st.selectbox("Survey preset", ["euclid", "lsst_y1", "lsst_y10"], key=f"{key}_hub_preset")
            ell_str = st.text_input("ℓ array (comma-separated)", "100, 300, 1000, 3000", key=f"{key}_hub_ell")
            delta_ell = st.number_input("Δℓ (bin width)", 50, 500, 200, key=f"{key}_hub_dell")
        with col2:
            om_syn = st.number_input("True Ωₘ", 0.1, 0.5, 0.3055, key=f"{key}_hub_om")
            h0_syn = st.number_input("True H₀", 50.0, 80.0, 67.97, key=f"{key}_hub_h0")
            seed = st.number_input("Seed", 0, 9999, 42, key=f"{key}_hub_seed")
        
        if st.button("Generate Synthetic", type="primary", key=f"{key}_hub_gen"):
            try:
                from pramana.core.weak_lensing import make_synthetic_shear_dataset
                ell_array = np.array([float(x.strip()) for x in ell_str.split(",")])
                result = make_synthetic_shear_dataset(
                    preset, ell_array, float(delta_ell), Om=om_syn, H0=h0_syn, seed=int(seed)
                )
                result["source"] = "synthetic"
                st.session_state[key] = result
                st.success(f"Generated synthetic shear data ({len(result['pairs'])} bin pairs, "
                          f"{len(ell_array)} ℓ bins)")
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error generating synthetic shear: {e}")


def dataset_loader(family_id: str, key: str | None = None, show_instructions: bool = True) -> dict | None:
    """Unified loader for one dataset family. Returns the loaded dict (or
    None if nothing loaded yet). Stores result in st.session_state[key]."""
    if family_id not in DATASET_CATALOG:
        raise ValueError(f"Unknown dataset family: {family_id}")
    spec = DATASET_CATALOG[family_id]
    if key is None:
        key = spec["default_key"]

    st.markdown(f"### {spec['title']}")
    st.caption(spec["description"])

    if show_instructions:
        with st.expander("ℹ️ What is this dataset?", expanded=False):
            st.markdown(
                f"{spec['description']}\n\n**Official sources:**\n" +
                "\n".join(f"- [{s['label']}]({s['url']})" for s in spec["sources"]) +
                f"\n\n**Expected files:** {spec['files']}"
            )

    st.markdown(f"**Status:** {_family_status(key)}")

    tab_dl, tab_path, tab_syn = st.tabs(["📥 Download & Upload", "📂 File Path", "🧪 Synthetic"])
    with tab_dl:
        _render_download_tab(family_id, key)
    with tab_path:
        _render_path_tab(family_id, key)
    with tab_syn:
        _render_synthetic_tab(family_id, key)

    st.markdown("---")
    return st.session_state.get(key)


# ---------------------------------------------------------------------
# Renderers for a loaded family (used by Data Explorer)
# ---------------------------------------------------------------------

def render_pantheon_summary(key: str = "pantheon_data"):
    if key not in st.session_state:
        return
    from pramana.web.components.data_loader import render_data_summary
    render_data_summary(key)


def render_bao_summary(key: str = "bao_data"):
    if key not in st.session_state:
        return
    d = st.session_state[key]
    import plotly.graph_objects as go
    from pramana.web.components.ui import plotly_template, plot_export_controls

    st.markdown("**BAO Data Summary**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Data points", len(d["data"]))
    col2.metric("z range", f"{d['z_arr'].min():.3f} – {d['z_arr'].max():.3f}")
    col3.metric("Cov condition", f"{np.linalg.cond(d['cov']):.2e}")

    labels = [f"{t} {o}" for t, o in d["labels"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=d["data"], marker_color="#5FB3B3", name="data"))
    fig.update_layout(title="BAO data vector (DM/rd, DH/rd, DV/rd)",
                      height=320, template=plotly_template(), xaxis_tickangle=-45)
    plot_export_controls(fig, "bao_data_vector")
    st.plotly_chart(fig, use_container_width=True)


def render_cmb_summary(key: str = "cmb_data"):
    if key not in st.session_state:
        return
    d = st.session_state[key]
    import plotly.graph_objects as go
    from pramana.web.components.ui import plotly_template, plot_export_controls

    st.markdown("**CMB Spectra Summary**")
    col1, col2 = st.columns(2)
    col1.metric("ℓ range", f"{int(d['ell'].min())} – {int(d['ell'].max())}")
    col2.metric("Lensing ℓ range", f"{int(d['ell_kk'].min())} – {int(d['ell_kk'].max())}")

    fig = go.Figure()
    ell = d["ell"]
    fig.add_trace(go.Scatter(x=ell, y=d["cl_tt"] * ell * (ell + 1) / (2 * np.pi),
                             name="TT", line=dict(width=2, color="#5FB3B3")))
    fig.add_trace(go.Scatter(x=ell, y=d["cl_ee"] * ell * (ell + 1) / (2 * np.pi),
                             name="EE", line=dict(width=1.5, color="#e0a458")))
    fig.add_trace(go.Scatter(x=ell, y=d["cl_te"] * ell * (ell + 1) / (2 * np.pi),
                             name="TE", line=dict(width=1.5, color="#8ab4f8")))
    fig.update_layout(title="CMB spectra (D_ell)", xaxis_title="ℓ",
                      yaxis_title="D_ℓ [μK²]", height=360, template=plotly_template())
    plot_export_controls(fig, "cmb_D_ell_spectra")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["ell_kk"], y=d["cl_kk"], name="κκ",
                              line=dict(width=2, color="#5FB3B3")))
    fig2.update_layout(title="Lensing convergence C_ℓ^κκ", xaxis_title="ℓ",
                       yaxis_title="C_ℓ^κκ", height=260, template=plotly_template())
    plot_export_controls(fig2, "cmb_lensing_spectrum")
    st.plotly_chart(fig2, use_container_width=True)


def render_jwst_summary(key: str = "jwst_data"):
    if key not in st.session_state:
        return
    d = st.session_state[key]
    from pramana.web.components.ui import plotly_template, plot_export_controls, export_downloads
    import pandas as pd

    st.markdown("**JWST-era Probes Summary**")

    col1, col2, col3 = st.columns(3)
    col1.metric("H0 measurements", len(d.get("h0_table", {})))
    col2.metric("S8 measurements", len(d.get("s8_table", {})))
    col3.metric("High-z SNe", len(d.get("highz_sn", {}).get("z", [])))

    if d.get("h0_table"):
        h0_df = pd.DataFrame([
            {"Measurement": k, "H₀": v["H0"], "Error": v["err"], "Family": v["family"]}
            for k, v in d["h0_table"].items()
        ])
        st.dataframe(h0_df, use_container_width=True, hide_index=True)
        export_downloads(h0_df, "jwst_H0_measurements")

    if d.get("s8_table"):
        s8_df = pd.DataFrame([
            {"Measurement": k, "S₈": v["S8"], "Error": v["err"], "Family": v["family"]}
            for k, v in d["s8_table"].items()
        ])
        st.dataframe(s8_df, use_container_width=True, hide_index=True)
        export_downloads(s8_df, "jwst_S8_measurements")

    if d.get("highz_sn"):
        hz = d["highz_sn"]
        import plotly.graph_objects as go
        from pramana.web.components.ui import plotly_template, plot_export_controls
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hz["z"], y=hz["mb"], mode="markers",
                                 error_y=dict(type="data", array=hz["err"]),
                                 marker=dict(size=6, color="#e0a458"), name="High-z SNe"))
        fig.update_layout(title="High-z SNe (JWST-era)", xaxis_title="z",
                          yaxis_title="m_b", height=300, template=plotly_template())
        plot_export_controls(fig, "jwst_highz_SNe")
        st.plotly_chart(fig, use_container_width=True)
        
        highz_df = pd.DataFrame({"z": hz["z"], "mb": hz["mb"], "err": hz["err"]})
        export_downloads(highz_df, "jwst_highz_SNe")


def render_shear_summary(key: str = "shear_data"):
    if key not in st.session_state:
        return
    d = st.session_state[key]
    import plotly.graph_objects as go
    from pramana.web.components.ui import plotly_template, plot_export_controls

    st.markdown("**Weak Lensing (Cosmic Shear) Summary**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Survey", d.get("preset", "?"))
    col2.metric("Bin pairs", len(d.get("pairs", [])))
    col3.metric("ℓ bins", len(d.get("ell_array", [])))
    col4.metric("Source", d.get("source", "?"))

    if "data_vector" in d and "pairs" in d and "ell_array" in d:
        ell_array = d["ell_array"]
        pairs = d["pairs"]
        data_vec = d["data_vector"]
        n_pairs = len(pairs)
        n_ell = len(ell_array)
        
        fig = go.Figure()
        for idx, (i, j) in enumerate(pairs):
            vec = data_vec[idx * n_ell:(idx + 1) * n_ell]
            fig.add_trace(go.Scatter(
                x=ell_array, y=vec, mode="lines", name=f"({i},{j})",
                line=dict(width=1.5), opacity=0.8
            ))
        fig.update_layout(
            title="Shear C_ell (auto + cross spectra)", xaxis_title="ℓ",
            yaxis_title="C_ℓ", height=400, template=plotly_template(),
            xaxis_type="log", yaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True)
        plot_export_controls(fig, "wl_spectra")

    if "truth" in d:
        t = d["truth"]
        st.markdown(f"**Injected truth**: Ωₘ={t.get('Om', '?'):.4f}, H₀={t.get('H0', '?'):.2f}, A_IA={t.get('A_IA', 0.0):.2f}")


def render_family_summary(family_id: str, key: str | None = None):
    """Dispatch to the right summary renderer."""
    spec = DATASET_CATALOG[family_id]
    if key is None:
        key = spec["default_key"]
    return {
        "pantheon": render_pantheon_summary,
        "bao": render_bao_summary,
        "cmb": render_cmb_summary,
        "jwst": render_jwst_summary,
        "weak_lensing": render_shear_summary,
    }[family_id](key)
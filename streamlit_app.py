import os
import rasterio
import numpy as np
import streamlit as st
from typing import Dict, Optional, Tuple, List
import tempfile
from rasterio.warp import transform_bounds
import folium
from streamlit_folium import st_folium
import zipfile

st.set_page_config(
    page_title="Orthomosaic Band Merger",
    page_icon="🛰️",
    layout="wide",
)

# -----------------------------
# Core logic (adapted from your script)
# -----------------------------

def find_band_files(folder_path: str) -> Tuple[Dict[str, str], str]:
    """Identifies and returns file paths for each band based on their names.

    Looks for filenames containing the tokens: blue, green, red-edge ("red edge"), red, nir
    Case-insensitive; underscores/dashes treated as spaces.
    Returns a dict of band -> filepath and an example filename (first .tif seen).
    """
    band_keywords = {
        "blue": None,
        "green": None,
        "red": None,
        "red-edge": None,
        "nir": None,
    }

    first_filename = None
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".tif", ".tiff")):
            if first_filename is None:
                first_filename = file  # store the first filename found

            file_lower = file.lower().replace("_", " ").replace("-", " ")  # normalize

            if "blue" in file_lower:
                band_keywords["blue"] = os.path.join(folder_path, file)
            elif "green" in file_lower:
                band_keywords["green"] = os.path.join(folder_path, file)
            elif "red edge" in file_lower:  # detect "red edge" with space
                band_keywords["red-edge"] = os.path.join(folder_path, file)
            elif ("red" in file_lower) and ("red edge" not in file_lower):
                band_keywords["red"] = os.path.join(folder_path, file)
            elif "nir" in file_lower:
                band_keywords["nir"] = os.path.join(folder_path, file)

    missing_bands = [band for band, path in band_keywords.items() if path is None]
    if missing_bands:
        raise ValueError(f"Missing bands: {missing_bands}. Check file names in: {folder_path}")

    return band_keywords, first_filename or "example.tif"


def generate_output_filename(output_folder: str, flight_date: str, example_filename: str) -> str:
    """Generates a unique output filename based on the flight date and example file name."""
    parts = example_filename.split("_")
    base_name = "_".join(parts[:2]) if len(parts) >= 2 else os.path.splitext(example_filename)[0]
    composite_name = f"{base_name}_{flight_date}_composite.tif"
    return os.path.join(output_folder, composite_name)


def find_reflectance_folder(flight_path: str) -> Optional[str]:
    """Finds the reflectance folder inside a flight date directory.

    First tries: <flight>/<anySubfolder>/4_index/reflectance
    If not found, falls back to a recursive search for any folder named "reflectance".
    """
    # Try the original one-level logic
    for subfolder in os.listdir(flight_path):
        subfolder_path = os.path.join(flight_path, subfolder)
        if os.path.isdir(subfolder_path):
            reflectance_path = os.path.join(subfolder_path, "4_index", "reflectance")
            if os.path.exists(reflectance_path):
                return reflectance_path

    # Fallback: recursive search for any "reflectance" dir
    for root, dirs, _ in os.walk(flight_path):
        for d in dirs:
            if d.lower() == "reflectance":
                return os.path.join(root, d)

    return None


def merge_bands_from_folder(folder_path: str, output_folder: str, flight_date: str, overwrite: bool = False) -> Optional[str]:
    """Merges identified single-band orthomosaics from a folder into a 5-band composite orthomosaic."""
    band_files, example_filename = find_band_files(folder_path)

    ordered_bands = [
        band_files["blue"],
        band_files["green"],
        band_files["red"],
        band_files["red-edge"],
        band_files["nir"],
    ]

    output_path = generate_output_filename(output_folder, flight_date, example_filename)

    if (not overwrite) and os.path.exists(output_path):
        return output_path  # Skip writing; already exists

    # Base metadata from first band
    with rasterio.open(ordered_bands[0]) as src:
        meta = src.meta.copy()

    # Update metadata for multi-band float32 GeoTIFF
    meta.update(count=5, dtype=rasterio.float32)

    # Read each band
    bands: List[np.ndarray] = []
    for band_path in ordered_bands:
        with rasterio.open(band_path) as src:
            bands.append(src.read(1))

    bands_np = np.stack(bands, axis=0)

    # Write output
    os.makedirs(output_folder, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        for i in range(5):
            dst.write(bands_np[i].astype(np.float32), i + 1)

    return output_path


def process_all_flight_dates(base_folder: str, output_folder: str, overwrite: bool = False, status_area=None) -> List[str]:
    """Scans all flight-date folders and processes them to generate composite orthomosaics.
    Returns a list of successfully written (or found) composite paths.
    """
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    os.makedirs(output_folder, exist_ok=True)

    results = []
    flight_dates = [d for d in sorted(os.listdir(base_folder)) if os.path.isdir(os.path.join(base_folder, d))]

    progress = st.progress(0)
    total = len(flight_dates) if flight_dates else 1

    for idx, flight_date in enumerate(flight_dates, start=1):
        flight_path = os.path.join(base_folder, flight_date)
        try:
            if status_area:
                status_area.write(f"🔍 Searching reflectance in **{flight_path}**")

            reflectance_folder = find_reflectance_folder(flight_path)
            if not reflectance_folder:
                status_area.error(f"⚠️ Reflectance not found in: {flight_path}")
                progress.progress(min(idx / total, 1.0))
                continue

            if status_area:
                status_area.write(f"📂 Found: `{reflectance_folder}` → merging…")

            out_path = merge_bands_from_folder(reflectance_folder, output_folder, flight_date, overwrite=overwrite)
            results.append(out_path)
            status_area.success(f"✅ Composite ready: `{out_path}`")
        except Exception as e:
            status_area.error(f"❌ Error in {flight_path}: {e}")
        finally:
            progress.progress(min(idx / total, 1.0))

    return results


# -----------------------------
# Quicklook + Interactive Map
# -----------------------------

def _percentile_stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a[np.isfinite(a)], [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)


def make_rgb_quicklook_png(composite_path: str) -> str:
    """Create a temporary RGB PNG (R=band3, G=band2, B=band1) with simple percentile stretch.
    Returns the PNG path.
    """
    with rasterio.open(composite_path) as ds:
        # Our order is [B,G,R,RE,NIR] → indices 1..5
        b_red = ds.read(3)
        b_green = ds.read(2)
        b_blue = ds.read(1)
        # Stretch each
        R = _percentile_stretch(b_red)
        G = _percentile_stretch(b_green)
        B = _percentile_stretch(b_blue)
        rgb = np.dstack([R, G, B])

        # Save PNG to tmp
        tmpdir = tempfile.gettempdir()
        out_png = os.path.join(tmpdir, f"quicklook_{os.path.basename(composite_path)}.png")
        from imageio import imwrite
        imwrite(out_png, rgb)

        return out_png


def show_interactive_map(composite_path: str):
    """Render a zoomable Folium map with an ImageOverlay of the quicklook RGB.
    Reprojects bounds to EPSG:4326 for display.
    """
    with rasterio.open(composite_path) as ds:
        bounds = ds.bounds
        crs = ds.crs
        # Transform bounds to WGS84
        wgs84_bounds = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
        (minx, miny, maxx, maxy) = wgs84_bounds
        center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]

    png_path = make_rgb_quicklook_png(composite_path)

    m = folium.Map(location=center, zoom_start=16, control_scale=True, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        name=os.path.basename(composite_path),
        image=png_path,
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    folium.LayerControl().add_to(m)

    st_folium(m, width=900, height=600)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛰️ Multispectral Orthomosaic Band Merger")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
        This app scans each *flight date* subfolder inside your **Base folder**, tries to find a
        `reflectance` directory (e.g., `.../<any>/4_index/reflectance`), then merges single-band
        orthomosaics (Blue, Green, Red, Red-Edge, NIR) into a 5‑band composite GeoTIFF per flight.

        **Band detection** is based on filenames containing these tokens (case-insensitive):
        `blue`, `green`, `red edge`, `red`, `nir`.
        """
    )

st.sidebar.header("⚙️ Settings")
base_folder = st.sidebar.text_input(
    "Base folder (contains flight-date folders)",
    value="",
)
output_folder = st.sidebar.text_input(
    "Output folder (will contain composites)",
    value="",
)
overwrite = st.sidebar.checkbox("Overwrite existing composites", value=False)

# --- Upload ZIP (for Streamlit Cloud / sharing) ---
st.sidebar.markdown("### 📦 Upload a ZIP (Cloud-friendly)")
zip_file = st.sidebar.file_uploader(
    "Upload a .zip containing your `FLIGHT DATES` tree (or a folder with `reflectance`)",
    type=["zip"],
)

if zip_file is not None:
    tmp_root = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_root, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_root)
    st.sidebar.success(f"ZIP extracted to: {tmp_root}")

    # Try to auto-detect a 'FLIGHT DATES' directory
    candidates = []
    for root, dirs, files in os.walk(tmp_root):
        for d in dirs:
            if d.lower().strip() == "flight dates":
                candidates.append(os.path.join(root, d))

    if candidates:
        base_folder = candidates[0]
        st.sidebar.info(f"Base folder set to: {base_folder}")
    else:
        st.sidebar.warning("No 'FLIGHT DATES' dir found; set Base folder manually in sidebar.")

    # Default output inside temp root
    output_folder = os.path.join(tmp_root, "ORTHOMOSAICS")
    st.sidebar.info(f"Output folder set to: {output_folder}")

col_run, col_preview = st.columns([1, 1])

with col_run:
    st.subheader("Run")
    status_area = st.empty()
    if st.button("🚀 Process all flight dates", type="primary"):
        try:
            results = process_all_flight_dates(base_folder, output_folder, overwrite=overwrite, status_area=status_area)
            if results:
                st.success(f"🎉 Done! {len(results)} composites ready.")
            else:
                st.warning("No composites created. Check folders and names.")
        except Exception as e:
            st.error(f"Failed: {e}")

with col_preview:
    st.subheader("Preview outputs")
    if os.path.isdir(output_folder):
        composites = [f for f in sorted(os.listdir(output_folder)) if f.lower().endswith((".tif", ".tiff"))]
        st.write(f"Found **{len(composites)}** file(s) in output folder.")
        if composites:
            sel = st.selectbox("Choose a composite to inspect filename", composites)
            if sel:
                sel_path = os.path.join(output_folder, sel)
                st.code(sel_path)
                try:
                    with rasterio.open(sel_path) as ds:
                        st.write({
                            "width": ds.width,
                            "height": ds.height,
                            "count": ds.count,
                            "crs": str(ds.crs),
                            "dtype": ds.dtypes,
                        })
                except Exception as e:
                    st.warning(f"Could not open with rasterio: {e}")

                st.divider()
                st.write("### 🔍 Interactive map preview (RGB quicklook)")
                if st.button("Show map for selected composite"):
                    try:
                        show_interactive_map(sel_path)
                    except Exception as e:
                        st.error(f"Map preview failed: {e}")
    else:
        st.info("Output folder does not exist yet.")

st.caption("Tip: on Streamlit Cloud, upload a ZIP or include sample data in the repo. Install once: `pip install streamlit-folium folium imageio`.")

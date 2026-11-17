"""Maps of 2025 crash locations, in the spirit of the KSI and High Injury
Network maps in the Safe Streets Lincoln Vision Zero Action Plan.

Outputs (in figures/):
  ksi_map.png             every crash as a faint dot, KSI crashes on top
                          (fatal vs. serious injury)
  crash_hotspots.png      crash count per location (LOCKEY), top 10 labeled
  crashes_by_street.png   top 15 streets by crash count, with KSI count
  crashes_by_street.csv   the table behind that chart
  top_25_locations.csv    the 25 locations (LOCKEYs) with the most crashes
  crash_map.html          interactive map, opened in a browser:
                          - layers: crash heat map, every crash as a dot,
                            KSI markers (details on click), top 25 locations
                            and High Risk Network tiers
                          - filters by severity, mode, date (month, week,
                            day), light and time of day, each option with a
                            live count and an "only" shortcut; they apply to
                            every layer but the tiers (a property of the
                            road, not of when crashes happened)
                          - a pinned status bar: crashes shown, active
                            filters, Reset filters and Reset view
                          - the top 25 is re-ranked for the filters and
                            listed in the side panel (hovering a row
                            highlights its bubble); "Rank by" swaps in the
                            traffic-adjusted top 25 from adt_rates.py, which
                            covers the whole year, marks hidden-risk
                            intersections and explains "expected"
                          - search jumps to any location (with its
                            traffic-adjusted rank, if rated); pop-ups, or a
                            click anywhere, link to Google Street View
                          - "About this map" notes the data and caveats

Coordinates: about 2% are blank, two Latitude cells hold case numbers, and a
few rows have Latitude and Longitude swapped. Swapped pairs are fixed, then
anything outside the Lincoln area is dropped.

LOCKEY 2 ("ALLEY & ALL") pools alley crashes from all over the city, so it
is left out of the location ranking (its crashes still appear as points).

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib --with contextily --with folium --with scipy python analysis/crash_map.py
"""

import json
import re
from pathlib import Path

import contextily as cx
import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import HeatMap

from adt_rates import traffic_adjusted
from high_risk_network import CONTROL_PTS, ROAD_CLASS_PTS, TIERS, adt_pts, tier
from ksi_by_gender_mode import MODE_ORDER, mode_and_gender

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

LAT_RANGE = (40.6, 41.0)
LON_RANGE = (-96.95, -96.45)
BASEMAP = cx.providers.Esri.WorldGrayCanvas
ALL_ALLEYS = 2
GRAY, SERIOUS, FATAL = "#a3a29d", "#eb6834", "#7a1c0c"
# Single-crash dots on the interactive map. Blue (heat map, top 25), orange/red
# (KSI) and purple (HRN tiers) are taken, and gray vanishes on the basemap.
DOT = "#2b9348"
SIDE_WIDTH = 280  # px, the left column of legend, layers and filters
INK, MUTED = "#0b0b0b", "#52514e"
HEAT_RADIUS, HEAT_BLUR, HEAT_MAX_ZOOM = 8, 10, 12
HEAT_GRADIENT = {0.2: "#2c7bb6", 0.4: "#00a6ca", 0.6: "#90eb9d", 0.8: "#f9d057", 1.0: "#d7191c"}
# Google Maps URL that opens the nearest Street View panorama; no API key needed.
STREET_VIEW = "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint="
LABEL_ALL_ZOOM = 13  # below this zoom, only ranks 1-10 are labeled
TIER_COLORS = {"High": "#5b2a86", "Moderate": "#9d7cc9", "Low": "#d6cbe6"}
# A darker rim per tier keeps the pale Low dots visible on the gray basemap.
TIER_EDGES = {"High": "#3b1659", "Moderate": "#6c4a9e", "Low": "#8e78b3"}
# The text column is blank in 489 rows, so names come from the code.
ROAD_CLASS_NAMES = {
    0: "Local", 1: "Major", 2: "Collector", 4: "Expressway", 11: "Major/Major",
    12: "Major/Collector", 13: "Major/Local", 22: "Collector/Collector",
    23: "Collector/Local", 33: "Local/Local",
}


def nice(name):
    if not isinstance(name, str):
        return "Unnamed location"
    words = re.sub(r"[\w'-]+", lambda w: w[0].capitalize(), name)
    return words.replace(" To ", " to ")


def load():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    df["CASE"] = range(1, len(df) + 1)
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    swapped = lat < 0
    df["lat"] = lat.where(~swapped, lon)
    df["lon"] = lon.where(~swapped, lat)
    df["fatal"] = df["# of Fatalities (1)"] > 0
    df["ksi"] = df["fatal"] | (df["# of Suspected Serious Injuries (2)"] > 0)
    df["location"] = df["On Street (Name)"].str.strip() + " & " + df["At Street"].astype(str).str.strip()
    midblock = df["Between Street \n(if midblock)"].notna()
    df["segment"] = midblock
    # High Risk Network score, as in high_risk_network.py (constant per LOCKEY).
    df["hrn_score"] = (
        df["Road Classification Code"].map(ROAD_CLASS_PTS).fillna(0)
        + df["ADT"].map(adt_pts)
        + df["Lockey Traffic Control Code"].map(CONTROL_PTS).fillna(0)
    ).astype(int)
    df["hrn_tier"] = df["hrn_score"].map(tier)
    df.loc[midblock, "location"] = (
        df["On Street (Name)"].str.strip() + " (" + df["At Street"].astype(str).str.strip()
        + " to " + df["Between Street \n(if midblock)"].astype(str).str.strip() + ")"
    )
    in_lincoln = df["lat"].between(*LAT_RANGE) & df["lon"].between(*LON_RANGE)
    print(f"Crashes: {len(df)}, mapped: {in_lincoln.sum()} "
          f"({swapped.sum()} lat/lon swaps fixed, {(~in_lincoln).sum()} dropped)")
    return df, df[in_lincoln].copy()


def style(ax, title, note):
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.figure.text(0.01, 0.01, note, fontsize=6.5, color=MUTED)


def add_basemap(ax):
    cx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, attribution_size=5)


def ksi_map(m):
    ksi = m[m["ksi"]]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.scatter(m["lon"], m["lat"], s=2, color=GRAY, alpha=0.35, linewidths=0,
               label=f"All crashes (n={len(m):,})")
    ax.scatter(ksi.loc[~ksi["fatal"], "lon"], ksi.loc[~ksi["fatal"], "lat"], s=30,
               color=SERIOUS, edgecolors="white", linewidths=0.8,
               label=f"Serious injury (n={(~ksi['fatal']).sum()})")
    ax.scatter(ksi.loc[ksi["fatal"], "lon"], ksi.loc[ksi["fatal"], "lat"], s=60, marker="D",
               color=FATAL, edgecolors="white", linewidths=0.8,
               label=f"Fatal (n={ksi['fatal'].sum()})")
    add_basemap(ax)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
              fontsize=8, markerscale=1.2)
    style(ax, "Where fatal and serious injury (KSI) crashes happened in Lincoln, 2025",
          "KSI = at least one fatality or suspected serious injury. Basemap: Esri World Gray Canvas.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "ksi_map.png", facecolor="white")
    plt.close(fig)


def location_counts(m):
    return (
        m[m["LOCKEY"] != ALL_ALLEYS].groupby("LOCKEY")
        .agg(location=("location", "first"), lat=("lat", "median"), lon=("lon", "median"),
             crashes=("ksi", "size"), ksi=("ksi", "sum"), segment=("segment", "first"),
             adt=("ADT", "first"), hrn_score=("hrn_score", "first"), hrn_tier=("hrn_tier", "first"))
        .sort_values("crashes", ascending=False)
    )


def top_locations_csv(locs):
    top = locs.head(25).reset_index()
    pd.DataFrame({
        "rank": range(1, len(top) + 1),
        "location": top["location"].map(nice),
        "type": top["segment"].map({True: "Road segment", False: "Intersection"}),
        "crashes": top["crashes"],
        "ksi_crashes": top["ksi"],
        "adt": top["adt"].astype("Int64"),
        "hrn_tier": top["hrn_tier"],
        "hrn_score": top["hrn_score"],
        "lockey": top["LOCKEY"],
        "lat": top["lat"].round(5),
        "lon": top["lon"].round(5),
    }).to_csv(FIG / "top_25_locations.csv", index=False)


def hotspot_map(locs):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.scatter(locs["lon"], locs["lat"], s=locs["crashes"] * 4, color="#2a78d6",
               alpha=0.45, edgecolors="white", linewidths=0.5)
    top = locs.head(10)
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        ax.annotate(str(rank), (r["lon"], r["lat"]), fontsize=7, fontweight="bold",
                    color=INK, ha="center", va="center")
    add_basemap(ax)
    for s in (5, 20, 50):
        ax.scatter([], [], s=s * 4, color="#2a78d6", alpha=0.45, label=f"{s} crashes")
    key = "\n".join(f"{i}. {nice(r['location'])} ({r['crashes']})"
                    for i, (_, r) in enumerate(top.iterrows(), 1))
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
              fontsize=7, labelspacing=1.4, borderpad=1)
    ax.text(0.99, 0.01, key, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    style(ax, "Crash hotspots in Lincoln, 2025: crashes per intersection or segment",
          "Circle area is proportional to the number of crashes at each location (LOCKEY). "
          "Basemap: Esri World Gray Canvas.")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "crash_hotspots.png", facecolor="white")
    plt.close(fig)


def street_chart(df):
    streets = (
        df.assign(street=df["On Street (Name)"].str.strip())
        .groupby("street").agg(crashes=("ksi", "size"), ksi=("ksi", "sum"))
        .sort_values("crashes", ascending=False).head(15)
    )
    streets.to_csv(FIG / "crashes_by_street.csv")

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    y = range(len(streets))
    bars = ax.barh(y, streets["crashes"], height=0.7, color="#2a78d6")
    ax.bar_label(bars, labels=[f"{c}  ({k} KSI)" if k else str(c)
                               for c, k in zip(streets["crashes"], streets["ksi"])],
                 padding=3, fontsize=8, color=MUTED)
    ax.set_yticks(list(y), [nice(s) for s in streets.index])
    ax.invert_yaxis()
    ax.set_xlabel("Number of crashes", color=MUTED)
    ax.set_xlim(0, streets["crashes"].max() * 1.2)
    ax.set_title("Streets with the most crashes in Lincoln, 2025", loc="left", fontsize=11, color=INK)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0)
    fig.text(0.01, 0.01, "Street = the 'On Street' of the crash location, so a crash at an "
             "intersection counts toward one street only. KSI = fatal or suspected serious injury.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "crashes_by_street.png", facecolor="white")
    plt.close(fig)
    return streets


# Categories for the filters. Each crash carries an index into these lists.
SEVERITY = ["Fatal or serious (KSI)", "Minor or possible injury", "No injury reported"]
LIGHT_NAMES = {1: "Daylight", 2: "Dawn or dusk", 3: "Dark, street lit", 4: "Dark, unlit"}
LIGHT_ORDER = [*LIGHT_NAMES.values(), "Unknown"]
TIME_BINS = [(0, 6, "12–6 am"), (6, 9, "6–9 am (rush)"), (9, 15, "9 am–3 pm"),
             (15, 19, "3–7 pm (rush)"), (19, 24, "7 pm–12 am")]
TIME_ORDER = [label for *_, label in TIME_BINS] + ["Unknown"]
# Row layout of the crash list sent to the browser; GROUPS name the filter fields.
#   [lat, lon, mode, day, place, LOCKEY, KSI 0/1, severity, light, time]
GROUPS = [
    dict(id="severity", title="Severity", field=7, options=SEVERITY),
    dict(id="mode", title="Mode", field=2, options=MODE_ORDER),
    dict(id="light", title="Light", field=8, options=LIGHT_ORDER),
    dict(id="time", title="Time of day", field=9, options=TIME_ORDER),
]


def severity_index(m):
    """0 KSI, 1 minor or possible injury only, 2 no injury reported."""
    minor = m[["# of Suspected Minor Injuries (3)", "# of Possible Injuries (4)"]].fillna(0).sum(axis=1)
    return (minor > 0).map({True: 1, False: 2}).where(~m["ksi"], 0)


def time_index(t):
    if pd.isna(t) or t == 9999:
        return len(TIME_BINS)
    hour = int(t) // 100
    return next((i for i, (lo, hi, _) in enumerate(TIME_BINS) if lo <= hour < hi), len(TIME_BINS))


def interactive_map(m, locs, n_total):
    year = int(pd.to_datetime(m["LPD Date"]).dt.year.mode()[0])
    # Traffic-adjusted ranking for the "Rank by" switch (top 25, as [LOCKEY,
    # crashes, KSI, expected, control, group]) and for search pop-ups (every
    # rated intersection, as LOCKEY -> [expected, rank, control]).
    adj, _ = traffic_adjusted(m, locs)
    adjusted = [[int(key), int(r["crashes"]), int(r["ksi"]), round(r["expected"], 1),
                 r["control"].lower(), r["group"]] for key, r in adj.head(25).iterrows()]
    rated = {int(key): [round(r["expected"], 1), int(r["excess_rank"]), r["control"].lower()]
             for key, r in adj.iterrows()}
    # No +/- buttons, so the side panel sits flush left; scroll and pinch still zoom.
    fmap = folium.Map(location=[m["lat"].median(), m["lon"].median()], zoom_start=12, tiles=None,
                      zoom_control=False)
    folium.TileLayer(BASEMAP, control=False).add_to(fmap)

    # One-hue blue ramp (orange is reserved for KSI markers). A low max_zoom keeps
    # point intensity from fading out at the city-wide view.
    # Starts with every crash; filter_js redraws it for the selected filters.
    heat = HeatMap(m[["lat", "lon"]].values.tolist(), name="All crashes (heat map)",
                   radius=HEAT_RADIUS, blur=HEAT_BLUR, max_zoom=HEAT_MAX_ZOOM, min_opacity=0.3,
                   gradient=HEAT_GRADIENT)
    heat.add_to(fmap)

    # KSI markers get their own pane so they stay above the location layers
    # however the layers are toggled.
    folium.map.CustomPane("ksi", z_index=650).add_to(fmap)

    # The tier belongs to the road, not to when crashes happened, so this layer
    # ignores the filters and its counts are for the whole year.
    hrn = folium.FeatureGroup(name="High Risk Network tier", show=False)
    for _, r in locs.sort_values("hrn_score").iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4, weight=1, color=TIER_EDGES[r["hrn_tier"]],
            fill=True, fill_opacity=0.9, fill_color=TIER_COLORS[r["hrn_tier"]],
            tooltip=(f"{nice(r['location'])}<br>{r['hrn_tier']} risk (score {r['hrn_score']}/8)"
                     f"<br>{r['crashes']} crashes, {r['ksi']} KSI in {year}"),
        ).add_to(hrn)
    hrn.add_to(fmap)

    # Drawn in the browser from the filtered crashes.
    hot = folium.FeatureGroup(name="Top 25 crash locations", show=False)
    hot.add_to(fmap)
    fmap.get_root().header.add_child(folium.Element(
        "<style>.few-labels .rank.minor { display: none; }</style>"))
    # The heat layer's maxZoom is a rendering setting, but Leaflet's layer control
    # reads it as a visibility limit and greys out the checkbox past it.
    fmap.get_root().script.add_child(folium.Element(
        "L.Control.Layers.prototype._checkDisabledLayers = () => {};"
        f"addEventListener('load', () => {{ const m = {fmap.get_name()}, few = () => m.getContainer()"
        f".classList.toggle('few-labels', m.getZoom() < {LABEL_ALL_ZOOM}); m.on('zoomend', few); few(); }});"))

    # Filled in the browser from the crash list, so the filters can redraw it.
    dots = folium.FeatureGroup(name="All crashes (dots)", show=False)
    dots.add_to(fmap)

    mode = m.apply(lambda row: mode_and_gender(row)[0], axis=1)
    m = m.assign(mode=mode, mode_i=mode.map(MODE_ORDER.index),
                 day=pd.to_datetime(m["LPD Date"]).dt.strftime("%Y-%m-%d"),
                 sev_i=severity_index(m),
                 light_i=m["Light Condition Code"].map(
                     {code: i for i, code in enumerate(LIGHT_NAMES)}).fillna(len(LIGHT_NAMES)).astype(int),
                 time_i=m["Time of Accident"].map(time_index))
    ksi = m[m["ksi"]]
    ksi_layer = folium.FeatureGroup(name="KSI crashes")
    for _, r in ksi.iterrows():
        ksi_layer.add_child(ksi_marker(r))
    ksi_layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    # Open framed on the crashes (middle 99.6%), clear of the side panel; Reset
    # view returns to the same framing.
    home = [[m["lat"].quantile(0.002), m["lon"].quantile(0.002)],
            [m["lat"].quantile(0.998), m["lon"].quantile(0.998)]]
    fmap.fit_bounds(home, padding_top_left=(SIDE_WIDTH + 24, 0))
    # One scrolling column flush left; the status bar stays pinned at its top.
    fmap.get_root().html.add_child(folium.Element(
        side_css() + '<div id="side">' + status_html() + legend_html(m, ksi, year) + search_html()
        + '<div id="layers-slot"></div>' + top_list_html(len(adj), year) + filters_html(year)
        + about_html(m, n_total, year) + "</div>"))
    layers = dict(MAP=fmap, HEAT=heat, HOT=hot, DOTS=dots, KSI=ksi_layer)
    fmap.get_root().script.add_child(folium.Element(filter_js(m, locs, adjusted, year, home, layers)))
    fmap.get_root().script.add_child(folium.Element(search_js(fmap, locs, rated, year, home)))
    fmap.get_root().script.add_child(folium.Element(street_view_js(fmap)))
    fmap.save(FIG / "crash_map.html")


def street_view(lat, lon):
    return f'<a href="{STREET_VIEW}{lat},{lon}" target="_blank" rel="noopener">Street View ↗</a>'


def street_view_js(fmap):
    """streetView() for the popups built in the browser, and a Street View link
    wherever the map itself is clicked. Markers with their own popup don't pass
    clicks through to the map."""
    return f"""
const streetView = (lat, lon) => `{street_view("${lat}", "${lon}")}`;
addEventListener('load', () => {{
  const map = {fmap.get_name()};
  map.on('click', e => L.popup().setLatLng(e.latlng)
    .setContent(streetView(e.latlng.lat.toFixed(5), e.latlng.lng.toFixed(5)).replace('Street View', 'Street View here'))
    .openOn(map));
}});"""


def fill_template(template, **values):
    """Replace __NAME__ tokens, so the JavaScript needs no doubled braces."""
    for name, value in values.items():
        template = template.replace(f"__{name}__", value)
    return template


def side_css():
    """The left column and the panel look shared by everything in it."""
    return f"""
    <style>
      #side {{ position:fixed;top:12px;left:12px;bottom:12px;width:{SIDE_WIDTH}px;z-index:1000;
               display:flex;flex-direction:column;gap:10px;overflow-y:auto;padding:2px }}
      #side .panel, #side .leaflet-control-layers {{
        background:rgba(255,255,255,.96) !important;padding:10px 14px !important;border-radius:6px !important;
        border:none !important;box-shadow:0 1px 4px rgba(0,0,0,.25) !important;margin:0 !important;
        font:13px/1.5 sans-serif;color:{INK};flex:none }}
      #side .leaflet-control-layers-overlays::before {{ content:"Layers";display:block;font-weight:700 }}
      #side .leaflet-control-layers label {{ margin:1px 0;cursor:pointer }}
      #side .note {{ color:{MUTED};font-size:11px;margin-bottom:4px }}
      #side button {{ font:11px sans-serif;cursor:pointer }}
      #side button:disabled {{ cursor:default;opacity:.45 }}
      #side summary {{ list-style:none;cursor:pointer }}
      #side summary::-webkit-details-marker {{ display:none }}
      #side summary::before {{ content:"▸ ";flex:none }}
      #side details[open] > summary::before {{ content:"▾ " }}
      #side #status {{ position:sticky;top:-2px;z-index:900;border-left:3px solid #2a78d6 !important;background:#fff !important }}
      #status .btns {{ display:flex;gap:6px;margin-top:6px }}
      #status button {{ flex:1;font-size:12px;padding:3px 6px }}
      .filter .h {{ font-weight:700;display:flex;justify-content:space-between;align-items:center;gap:6px }}
      .filter .state {{ font-weight:400;font-size:11px;color:#2a78d6 }}
      .filter .h > span:first-of-type {{ margin-right:auto }}
      .filter select {{ width:100%;margin-top:3px;font:13px sans-serif }}
      .filter select:disabled {{ display:none }}
      .filter label.opt {{ display:flex;align-items:baseline;gap:5px;cursor:pointer;line-height:1.6 }}
      .filter label.opt .n {{ color:{MUTED};font-size:11px;white-space:nowrap }}
      .filter label.opt .only {{ margin-left:auto;visibility:hidden;border:none;background:none;
                                 color:#2a78d6;padding:0 2px }}
      .filter label.opt:hover .only, .filter label.opt .only:focus {{ visibility:visible }}
      .filter input[type=search] {{ width:100%;box-sizing:border-box;margin-top:3px;font:13px sans-serif }}
      #top-rows {{ max-height:216px;overflow-y:auto;margin:0 -6px }}
      #top-rows .top-row {{ display:flex;gap:6px;padding:1px 6px;border-radius:4px;cursor:pointer;font-size:12px }}
      #top-rows .top-row b {{ width:20px;text-align:right;flex:none }}
      #top-rows .top-row .nm {{ flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap }}
      #top-rows .top-row .n {{ color:{MUTED};font-size:11px;white-space:nowrap }}
      #top-rows .top-row .tag {{ font-size:10px;color:#2a78d6;border:1px dashed #2a78d6;border-radius:3px;
                                 padding:0 3px;line-height:15px;align-self:center }}
      #top-rows .top-row.hl {{ background:#dbe8f8 }}
      .rank.hl {{ display:block !important;font-size:15px !important;color:#2a78d6 !important }}
      .key {{ display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin-right:4px;
              border:1px solid #2a78d6;box-sizing:border-box }}
      #about p {{ margin:6px 0 0 }}
    </style>"""


def status_html():
    return """
    <div id="status" class="panel">
      <div id="shown"></div>
      <div id="active" class="note" style="margin:2px 0 0"></div>
      <div class="btns">
        <button id="reset-all" type="button" title="Show every crash again">Reset filters</button>
        <button id="reset-view" type="button" title="Close pop-ups and zoom back out">Reset view</button>
      </div>
    </div>"""


def top_list_html(n_rated, year):
    return f"""
    <div id="top-list" class="panel filter" hidden>
      <div class="h" id="top-title">Top 25 locations</div>
      <label class="note" style="display:block;margin:2px 0 0">Rank by
        <select id="rank-by">
          <option value="count">Number of crashes</option>
          <option value="excess">Crashes above expected for its traffic</option>
        </select></label>
      <div class="note" id="top-note"></div>
      <div id="excess-info" class="note" hidden>
        <div><span class="key" style="background:rgba(42,120,214,.5)"></span>Also in the top 25 by crash count</div>
        <div><span class="key" style="background:rgba(42,120,214,.15);border-style:dashed;border-width:2px"></span>Hidden
          risk: not in the top 25 by crash count</div>
        <details style="margin-top:4px">
          <summary style="color:{INK}">How is "expected" worked out?</summary>
          <p style="margin:4px 0 0">Each intersection is compared with others that have the same kind of traffic
            control (signal, stop or yield, or roundabout) and similar traffic. "Expected" is the number of
            crashes a typical intersection like it had in {year}, from a model fit to all {n_rated:,} rated
            intersections.</p>
          <p style="margin:4px 0 0">The ranking uses each intersection's crashes above that expectation. A
            few crashes at a quiet intersection are mostly treated as chance, so small counts can't jump to
            the top. Only intersections with recorded traffic above 2,500 vehicles a day are rated.</p>
        </details>
      </div>
      <div id="top-rows"></div>
    </div>"""


def group_panel(gid, title, collapsed=False):
    """A checkbox filter; filter_js fills the options from GROUPS."""
    head = (f'<span>{title} <span class="state" id="{gid}-state"></span></span>'
            f'<button id="{gid}-reset" type="button">Reset</button>')
    body = f'<div id="{gid}-opts"></div>'
    if collapsed:
        return f'<details class="panel filter"><summary class="h">{head}</summary>{body}</details>'
    return f'<div class="panel filter"><div class="h">{head}</div>{body}</div>'


def filters_html(year):
    return (
        group_panel("severity", "Severity") + group_panel("mode", "Mode")
        + f"""
    <div id="date-filter" class="panel filter">
      <div class="h"><span>Date <span class="state" id="date-state"></span></span>
        <button id="date-reset" type="button">Reset</button></div>
      <select id="month"><option value="">All of {year}</option></select>
      <select id="week" disabled></select>
      <select id="day" disabled></select>
    </div>"""
        + group_panel("light", "Light", collapsed=True) + group_panel("time", "Time of day", collapsed=True))


def search_html():
    return """
    <div id="search-box" class="panel filter">
      <div class="h">Find a location <button id="search-reset" type="button">Clear</button></div>
      <div class="note">Street or intersection, e.g. "27th O"</div>
      <input id="search" type="search" list="places" placeholder="Search" autocomplete="off">
      <datalist id="places"></datalist>
      <div id="search-msg" class="note" style="margin:4px 0 0"></div>
    </div>"""


def about_html(m, n_total, year):
    t = pd.Timestamp.today()
    today = f"{t:%B} {t.day}, {t.year}"
    return f"""
    <details id="about" class="panel">
      <summary><b>About this map</b></summary>
      <div class="note" style="margin:0">
        <p><b>Data.</b> City of Lincoln crash records for January 1 to December 31, {year}: {n_total:,}
          crashes. {len(m):,} have usable coordinates and are mapped; the other {n_total - len(m):,} count in
          citywide totals but cannot be shown. The {year} file may not be final, since injury severity can be
          updated after a crash is reported.</p>
        <p><b>KSI</b> crashes are those in which at least one person was killed or suspected to be seriously
          injured.</p>
        <p><b>Every count on the map counts crashes</b>, not people. Filters apply to the heat map, dots,
          KSI markers and top 25 by crash count, but not to the risk tiers or the traffic-adjusted ranking,
          which describe the whole year.</p>
        <p><b>Risk tiers</b> adapt the Safe Streets Lincoln Vision Zero Action Plan's High Risk Network
          using the three risk factors in the crash data (road type, traffic and control or lanes).</p>
        <p><b>Traffic (ADT)</b> values come from the crash file. Which street they describe at an
          intersection is being confirmed with the City, and intersections with round values of 2,500 or
          less, which look like estimates, are not rated.</p>
        <p>Prepared by Cooper Malone, Linda Ngo and Sophia Thompson, MATH 435, University of
          Nebraska&ndash;Lincoln. Generated {today}.</p>
      </div>
    </details>"""


SEARCH_JS = r"""
addEventListener('load', () => {
  const map = __MAP__, places = __PLACES__, rated = __RATED__;
  const YEAR = __YEAR__, NRATED = __NRATED__, HOME = __HOME__, PAD = __PAD__;
  const $ = id => document.getElementById(id);
  $('places').innerHTML = places.map(p => `<option value="${p[0]}">`).join('');
  // An exact name first, else the busiest place whose name has every typed word,
  // so "27th o" finds "O St & 27th St".
  function find(q) {
    q = q.trim().toLowerCase();
    if (!q) return null;
    const words = q.split(/[\s&,]+/).filter(Boolean);
    return places.find(p => p[0].toLowerCase() === q)
      || places.find(p => words.every(w => p[0].toLowerCase().split(/[\s&()]+/).some(t => t.startsWith(w))));
  }
  function go() {
    const p = find($('search').value);
    if (!p) { $('search-msg').textContent = $('search').value.trim() ? 'No matching location' : ''; return; }
    const [name, lat, lon, crashes, ksi, kind, tier, key] = p;
    $('search').value = name;
    $('search-msg').textContent = '';
    map.flyTo([lat, lon], 16);
    const adj = rated[key];
    const extra = adj ? `<br>Expected about ${adj[0]} for a ${adj[2]} intersection with this traffic`
      + ` (#${adj[1]} of ${NRATED.toLocaleString()} by crashes above expected)` : '';
    L.popup().setLatLng([lat, lon]).setContent(
      `<b>${name}</b><br>${crashes} crash${crashes === 1 ? '' : 'es'}, ${ksi} KSI in ${YEAR}${extra}`
      + `<br>${kind} · ${tier} risk<br>${streetView(lat, lon)}`).openOn(map);
  }
  // Clear empties the search and zooms back out.
  $('search-reset').onclick = () => {
    $('search').value = '';
    $('search-msg').textContent = '';
    map.closePopup();
    map.flyToBounds(HOME, {paddingTopLeft: [PAD, 0]});
  };
  $('search').addEventListener('change', go);
  $('search').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
});"""


def search_js(fmap, locs, rated, year, home):
    # Every location as [name, lat, lon, crashes, ksi, kind, tier, LOCKEY], busiest
    # first. A name can cover more than one LOCKEY; the busiest one is kept.
    places = (locs.reset_index()
              .assign(name=lambda d: d["location"].map(nice),
                      kind=lambda d: d["segment"].map({True: "road segment", False: "intersection"}))
              .drop_duplicates("name")
              [["name", "lat", "lon", "crashes", "ksi", "kind", "hrn_tier", "LOCKEY"]]
              .round({"lat": 5, "lon": 5}).values.tolist())
    return fill_template(SEARCH_JS, MAP=fmap.get_name(), PLACES=json.dumps(places),
                         RATED=json.dumps(rated), YEAR=str(year), NRATED=str(len(rated)),
                         HOME=json.dumps(home), PAD=str(SIDE_WIDTH + 24))


FILTER_JS = r"""
addEventListener('load', () => {
  const map = __MAP__, heat = __HEAT__, hot = __HOT__, dots = __DOTS__, ksiLayer = __KSI__;
  const crashes = __CRASHES__, places = __PLACES__, adjusted = __ADJUSTED__, GROUPS = __GROUPS__;
  const YEAR = __YEAR__, HOME = __HOME__, PAD = __PAD__, BLUE = '#2a78d6';
  const G = Object.fromEntries(GROUPS.map(g => [g.id, g]));
  const $ = id => document.getElementById(id);
  const stop = e => { e.preventDefault(); e.stopPropagation(); };
  const fmt = (d, opts) => d.toLocaleDateString('en-US', {timeZone: 'UTC', ...opts});
  const iso = d => d.toISOString().slice(0, 10);
  const utc = (y, mo, d) => new Date(Date.UTC(y, mo, d));
  const plural = (n, word) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 'es'}`;
  const renderer = L.canvas();
  const markers = ksiLayer.getLayers();
  // Each KSI marker as a crash-shaped row, so one filter test serves both.
  const markerRows = markers.map(mk => {
    const [mode, day, light, time] = mk.options.tags;
    return [0, 0, mode, day, '', 0, 1, 0, light, time];
  });

  // The layer list joins the side column, under the search box.
  $('layers-slot').replaceWith(document.querySelector('.leaflet-control-layers'));
  L.DomEvent.disableClickPropagation($('side'));
  L.DomEvent.disableScrollPropagation($('side'));
  // The top 25 list shows only while its layer is on.
  map.on('overlayadd overlayremove', e => {
    if (e.layer === hot) $('top-list').hidden = e.type === 'overlayremove';
  });

  // Checkbox filters: a count beside each option (for the other filters' selection),
  // an "only" button to keep just that option, and a Reset for the group.
  for (const g of GROUPS) {
    const box = $(g.id + '-opts');
    box.innerHTML = g.options.map((o, i) =>
      `<label class="opt"><input type="checkbox" value="${i}" checked><span>${o}</span>`
      + `<span class="n"></span><button type="button" class="only" title="Show only ${o}">only</button></label>`
    ).join('');
    g.boxes = [...box.querySelectorAll('input')];
    g.counts = [...box.querySelectorAll('.n')];
    g.boxes.forEach(b => b.onchange = apply);
    box.querySelectorAll('.only').forEach((btn, i) => btn.onclick = e => {
      stop(e);
      g.boxes.forEach((b, j) => b.checked = j === i);
      apply();
    });
    $(g.id + '-reset').onclick = e => { stop(e); g.boxes.forEach(b => b.checked = true); apply(); };
  }
  const picked = g => new Set(g.boxes.filter(b => b.checked).map(b => +b.value));
  const passes = (row, sets, skip) => GROUPS.every(g => g === skip || sets.get(g).has(row[g.field]));

  // Date: a month, then optionally a week, then a day.
  for (let mo = 0; mo < 12; mo++)
    $('month').add(new Option(fmt(utc(YEAR, mo, 1), {month: 'long'}), mo));

  // Weeks run Sunday to Saturday, cut off at the month's edges.
  function weeks(mo) {
    const out = [], end = utc(YEAR, mo + 1, 0).getUTCDate();
    let start = 1;
    while (start <= end) {
      const stopDay = Math.min(end, start + 6 - utc(YEAR, mo, start).getUTCDay());
      out.push([start, stopDay]);
      start = stopDay + 1;
    }
    return out;
  }

  function fill(select, label, items) {
    select.length = 0;
    select.add(new Option(label, ''));
    items.forEach(([text, value]) => select.add(new Option(text, value)));
    select.disabled = false;
  }

  // The selected date range as [first, last] ISO days, or null for the whole year.
  function range() {
    const mo = $('month').value;
    if (mo === '') return null;
    if ($('day').value !== '') return [$('day').value, $('day').value];
    const [a, b] = $('week').value === '' ? [1, utc(YEAR, +mo + 1, 0).getUTCDate()]
                                          : $('week').value.split('-').map(Number);
    return [iso(utc(YEAR, +mo, a)), iso(utc(YEAR, +mo, b))];
  }

  function dateLabel() {
    const mo = $('month').value;
    if (mo === '') return '';
    const month = $('month').options[$('month').selectedIndex].text;
    if ($('day').value !== '') return $('day').options[$('day').selectedIndex].text;
    if ($('week').value !== '') return `${month.slice(0, 3)} ${$('week').options[$('week').selectedIndex].text.replace(/^\w+ /, '')}`;
    return month;
  }

  function resetDate() {
    $('month').value = '';
    $('week').disabled = $('day').disabled = true;
  }
  $('date-reset').onclick = e => { stop(e); resetDate(); apply(); };

  $('month').onchange = () => {
    const mo = $('month').value;
    $('day').disabled = true;
    if (mo === '') $('week').disabled = true;
    else fill($('week'), 'All weeks', weeks(+mo).map(([a, b]) => [
      a === b ? fmt(utc(YEAR, +mo, a), {month: 'short', day: 'numeric'})
              : `${fmt(utc(YEAR, +mo, a), {month: 'short', day: 'numeric'})} – ${b}`, `${a}-${b}`]));
    apply();
  };
  $('week').onchange = () => {
    const mo = +$('month').value, w = $('week').value;
    if (w === '') $('day').disabled = true;
    else {
      const [a, b] = w.split('-').map(Number), days = [];
      for (let d = a; d <= b; d++)
        days.push([fmt(utc(YEAR, mo, d), {weekday: 'short', month: 'short', day: 'numeric'}), iso(utc(YEAR, mo, d))]);
      fill($('day'), 'All days', days);
    }
    apply();
  };
  $('day').onchange = apply;

  // The top 25: re-ranked from the crashes shown, or the fixed traffic-adjusted list.
  function drawLocations(pts) {
    const byExcess = $('rank-by').value === 'excess';
    $('excess-info').hidden = !byExcess;
    let top;
    if (byExcess) {
      top = adjusted.map(([key, n, k, exp, control, group]) =>
        ({p: places[key], n, k, exp, control, hidden: group === 'Hidden risk'}));
      $('top-title').textContent = 'Top 25 above expected for their traffic';
      $('top-note').textContent = `All ${YEAR} crashes; the filters don't change this list · hover to find, click to zoom`;
    } else {
      const counts = new Map();
      for (const c of pts) {
        if (!places[c[5]]) continue;  // the pooled alley LOCKEY
        const n = counts.get(c[5]) || [0, 0];
        n[0]++; n[1] += c[6];
        counts.set(c[5], n);
      }
      top = [...counts].map(([key, [n, k]]) => ({p: places[key], n, k}))
        .sort((a, b) => b.n - a.n || a.p[7] - b.p[7]).slice(0, 25);
      $('top-title').textContent = `Top ${top.length} location${top.length === 1 ? '' : 's'}`;
      $('top-note').textContent = 'For the crashes shown on the map · hover to find, click to zoom';
    }

    hot.clearLayers();
    $('top-rows').innerHTML = top.map((t, i) =>
      `<div class="top-row"><b>${i + 1}</b><span class="nm" title="${t.p[0]}">${t.p[0]}</span>`
      + (t.hidden ? '<span class="tag" title="Not in the top 25 by crash count">hidden</span>' : '')
      + `<span class="n">${byExcess ? `${t.n} vs ${t.exp} exp.` : `${t.n} · ${t.k} KSI`}</span></div>`
    ).join('') || '<div class="note">No crashes match the filters</div>';
    const rowEls = [...$('top-rows').children];
    top.forEach((t, i) => {
      const {p, n, k} = t, rank = i + 1, radius = 4 + n / 4;
      const adt = p[4] > 0 ? `${p[4].toLocaleString()} vehicles/day` : 'traffic n/a';
      const why = byExcess
        ? `<br>Expected about ${t.exp} for a ${t.control} intersection with this traffic`
          + (t.hidden ? '<br><b>Hidden risk:</b> not in the top 25 by crash count' : '')
        : '';
      const tip = `#${rank} ${p[0]}: ${plural(n, 'crash')}, ${k} KSI${why}<br>${p[3]} · ${adt} · ${p[5]} risk`;
      const base = t.hidden ? {weight: 2, dashArray: '4 3', fillOpacity: 0.15} : {weight: 1, dashArray: null, fillOpacity: 0.5};
      const circle = L.circleMarker([p[1], p[2]], {
        radius, color: BLUE, fill: true, bubblingMouseEvents: false, ...base,
      }).bindTooltip(tip).bindPopup(`${tip}<br>${streetView(p[1], p[2])}`);
      const label = L.marker([p[1], p[2]], {icon: L.divIcon({iconSize: [0, 0], className: '', html:
        `<div class="rank${rank > 10 ? ' minor' : ''}" style="position:absolute;left:0;bottom:${Math.round(radius + 1)}px;`
        + 'transform:translateX(-50%);font:700 12px sans-serif;color:__INK__;'
        + `white-space:nowrap;text-shadow:0 0 2px #fff,0 0 2px #fff,0 0 3px #fff">#${rank}</div>`})}).bindTooltip(tip);
      hot.addLayer(circle);
      hot.addLayer(label);
      label.on('click', () => circle.openPopup());
      // A row and its bubble light each other up, so crowded labels can be told apart.
      const row = rowEls[i];
      const light = on => {
        row.classList.toggle('hl', on);
        circle.setStyle(on ? {...base, weight: base.weight + 2, fillOpacity: 0.85} : base);
        if (on) circle.bringToFront();
        label.setZIndexOffset(on ? 1000 : 0);
        label.getElement()?.querySelector('.rank').classList.toggle('hl', on);
      };
      row.onmouseenter = () => light(true);
      row.onmouseleave = () => light(false);
      circle.on('mouseover', () => { light(true); row.scrollIntoView({block: 'nearest'}); });
      circle.on('mouseout', () => light(false));
      row.onclick = () => {
        map.flyTo([p[1], p[2]], Math.max(map.getZoom(), 15));
        circle.openPopup();
      };
    });
  }

  function apply() {
    const r = range(), inRange = day => !r || (day >= r[0] && day <= r[1]);
    const sets = new Map(GROUPS.map(g => [g, picked(g)]));
    const inDates = crashes.filter(c => inRange(c[3]));
    // Each option's count follows every other filter, so it shows what ticking it adds.
    for (const g of GROUPS) {
      const n = g.options.map(() => 0), k = g.options.map(() => 0);
      for (const c of inDates) if (passes(c, sets, g)) { n[c[g.field]]++; k[c[g.field]] += c[6]; }
      // Severity's own KSI count would only repeat its first row.
      g.counts.forEach((el, i) => el.textContent = g.id === 'severity' ? n[i].toLocaleString()
                                                     : `${n[i].toLocaleString()} · ${k[i]} KSI`);
      const on = sets.get(g).size;
      $(g.id + '-state').textContent = on === g.options.length ? '' : `${on} of ${g.options.length}`;
    }
    $('date-state').textContent = dateLabel();

    const pts = inDates.filter(c => passes(c, sets));
    dots.clearLayers();
    pts.forEach(c => dots.addLayer(L.circleMarker([c[0], c[1]], {
      renderer, radius: 3.5, color: '#fff', weight: 0.6, fill: true, fillColor: '__DOT__', fillOpacity: 0.85,
    }).bindTooltip(`${c[4]}<br>${fmt(new Date(c[3]), {month: 'short', day: 'numeric', year: 'numeric'})}`
      + ` · ${G.mode.options[c[2]]}<br>${G.severity.options[c[7]]}`)));
    heat.setLatLngs(pts.map(c => [c[0], c[1]]));
    drawLocations(pts);
    ksiLayer.clearLayers();
    const shown = markers.filter((mk, i) => inRange(markerRows[i][3]) && passes(markerRows[i], sets));
    shown.forEach(mk => ksiLayer.addLayer(mk));

    $('shown').innerHTML = `Showing <b>${pts.length.toLocaleString()}</b> of ${plural(crashes.length, 'crash')}`
      + ` · <b>${shown.length}</b> KSI`;
    const active = GROUPS.filter(g => sets.get(g).size < g.options.length)
      .map(g => `${g.title} (${sets.get(g).size} of ${g.options.length})`);
    if (r) active.push(dateLabel());
    $('active').textContent = active.length ? `Filtered by ${active.join(' · ')}` : 'No filters: every mapped crash is shown';
    $('reset-all').disabled = !active.length;
  }

  // Reset filters leaves the view alone; Reset view leaves the filters alone.
  $('reset-all').onclick = () => {
    GROUPS.forEach(g => g.boxes.forEach(b => b.checked = true));
    resetDate();
    apply();
  };
  $('reset-view').onclick = () => {
    map.closePopup();
    map.flyToBounds(HOME, {paddingTopLeft: [PAD, 0]});
  };
  $('rank-by').onchange = apply;
  addEventListener('keydown', e => { if (e.key === 'Escape') map.closePopup(); });
  apply();
});"""


def filter_js(m, locs, adjusted, year, home, layers):
    # Every mapped crash as [lat, lon, mode, day, location, LOCKEY, KSI 0/1,
    # severity, light, time]; mode, severity, light and time index into GROUPS.
    crashes = (m.assign(place=m["location"].map(nice), k=m["ksi"].astype(int))
               [["lat", "lon", "mode_i", "day", "place", "LOCKEY", "k", "sev_i", "light_i", "time_i"]]
               .round({"lat": 5, "lon": 5}).values.tolist())
    # Every ranked location by LOCKEY as [name, lat, lon, kind, adt, tier, score,
    # full-year rank]; the rank breaks ties so a full-year top 25 matches the CSV.
    places = {
        int(key): [nice(r["location"]), round(r["lat"], 5), round(r["lon"], 5),
                   "road segment" if r["segment"] else "intersection",
                   int(r["adt"]) if pd.notna(r["adt"]) else 0, r["hrn_tier"], int(r["hrn_score"]), i]
        for i, (key, r) in enumerate(locs.iterrows())
    }
    return fill_template(
        FILTER_JS, **{name: layer.get_name() for name, layer in layers.items()},
        CRASHES=json.dumps(crashes), PLACES=json.dumps(places), ADJUSTED=json.dumps(adjusted),
        GROUPS=json.dumps(GROUPS), YEAR=str(year), HOME=json.dumps(home), PAD=str(SIDE_WIDTH + 24),
        INK=INK, DOT=DOT)


def ksi_marker(r):
    severity = "Fatal" if r["fatal"] else "Serious injury"
    date = pd.to_datetime(r["LPD Date"]).strftime("%b %d, %Y")
    t = int(r["Time of Accident"]) if pd.notna(r["Time of Accident"]) else 9999
    time = "time unknown" if t == 9999 else f"{t // 100:02d}:{t % 100:02d}"
    details = [
        f"<b>{severity}</b>", nice(r["location"]), f"{date}, {time}",
        f"Mode: {r['mode']}",
        f"Type: {r['Type of Accident'].capitalize()}",
    ]
    if r["mode"] == "Pedestrian" and pd.notna(r["Ped Accident Type Desc"]):
        details.append(f"Pedestrian action: {r['Ped Accident Type Desc'].capitalize()}")
    details += [
        f"Light: {r['Light Condition']}",
        f"Road: {ROAD_CLASS_NAMES.get(r['Road Classification Code'], 'Unknown')}",
        f"High Risk Network: {r['hrn_tier']} (score {r['hrn_score']}/8)",
        f"Case #{r['CASE']}",
        street_view(r["lat"], r["lon"]),
    ]
    return folium.CircleMarker(
        [r["lat"], r["lon"]], radius=8 if r["fatal"] else 6, pane="ksi",
        color="white", weight=1.5, fill=True, fill_opacity=0.95, bubbling_mouse_events=False,
        fill_color=FATAL if r["fatal"] else SERIOUS, tags=[int(r["mode_i"]), r["day"], int(r["light_i"]), int(r["time_i"])],
        tooltip=f"{severity} ({r['mode'].lower()}): {nice(r['location'])}",
        popup=folium.Popup("<br>".join(details), max_width=280),
    )


def legend_html(m, ksi, year):
    def dot(color, size, edge=None):
        rim = f"border:1px solid {edge}" if edge else "border:1.5px solid #fff;box-shadow:0 0 0 1px #ccc"
        return (f'<span style="display:inline-block;width:{size}px;height:{size}px;border-radius:50%;'
                f'background:{color};{rim};vertical-align:middle;margin-right:6px"></span>')

    ramp = ", ".join(HEAT_GRADIENT.values())
    tiers = "".join(f'<div>{dot(TIER_COLORS[t], 10, TIER_EDGES[t])}{t}</div>' for t in TIERS)
    return f"""
    <div class="panel">
      <div style="font-size:16px;font-weight:700">Crashes in Lincoln, {year}</div>
      <div style="color:{MUTED};font-size:12px;margin-bottom:6px">
        {len(m):,} crashes mapped · {len(ksi)} fatal or serious injury (KSI)</div>
      <div>{dot(FATAL, 16)}Fatal ({ksi['fatal'].sum()})</div>
      <div>{dot(SERIOUS, 12)}Serious injury ({(~ksi['fatal']).sum()})</div>
      <div style="margin-top:6px">Crash density (heat map)</div>
      <div style="height:8px;width:160px;border-radius:4px;background:linear-gradient(to right, {ramp})"></div>
      <div style="display:flex;justify-content:space-between;width:160px;font-size:11px;color:{MUTED}">
        <span>fewer</span><span>more</span></div>
      <div style="margin-top:6px">{dot(DOT, 10)}Single crash (dots layer)</div>
      <div style="margin-top:6px">High Risk Network tier</div>
      <div style="display:flex;gap:10px;font-size:12px">{tiers}</div>
      {hrn_explainer()}
      <div style="color:{MUTED};font-size:11px;margin-top:6px">Click a KSI dot for details.</div>
    </div>"""


def hrn_explainer():
    """How the High Risk Network score is built, as in high_risk_network.py."""
    rows = [
        ("Road type", "0–3", "Local 0 · Collector 1 · Major 2 · Major/Major or Expressway 3"),
        ("Traffic (vehicles/day)", "0–3", "under 5k 0 · 5k–15k 1 · 15k–30k 2 · 30k+ 3"),
        ("Intersection control", "0–2", "signal 2 · stop, yield or none 1 · roundabout 0"),
        ("Segment lanes", "0–2", "4+ lanes 2 · 2 lanes with left turn 1 · 2 lanes or one way 0"),
    ]
    items = "".join(f'<div style="margin-top:4px"><b>{name}</b> ({pts})<br>{desc}</div>'
                    for name, pts, desc in rows)
    return f"""
      <details style="font-size:11px;color:{MUTED};margin-top:4px">
        <summary style="cursor:pointer;color:{INK}">How is risk scored?</summary>
        <div style="margin-top:2px">Each location gets points for risk factors that are not
          its crash history, adapted from the Safe Streets Lincoln Vision Zero plan:</div>
        {items}
        <div style="margin-top:4px">An intersection scores on control, a road segment on lanes.
          Total out of 8: <b>High</b> 6–8 · <b>Moderate</b> 3–5 · <b>Low</b> 0–2.</div>
      </details>"""


def main():
    df, mapped = load()
    FIG.mkdir(exist_ok=True)
    locs = location_counts(mapped)
    ksi_map(mapped)
    top_locations_csv(locs)
    hotspot_map(locs)
    streets = street_chart(df)
    interactive_map(mapped, locs, len(df))
    print("\nTop 10 locations:\n" + locs.head(10)[["location", "crashes", "ksi"]].to_string())
    print("\nTop streets:\n" + streets.to_string())
    print("\nSaved ksi_map.png, crash_hotspots.png, crashes_by_street.png/.csv, "
          "top_25_locations.csv, crash_map.html in figures/")


if __name__ == "__main__":
    main()

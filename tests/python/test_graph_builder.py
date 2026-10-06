"""Unit tests for src.data.graph_builder.

Every test here runs offline against a synthetic graph shaped like OSMnx
output. The one test that actually queries Overpass is marked ``network`` and
deselected by default (see pyproject.toml addopts); run it explicitly with:

    pytest -m network tests/python/test_graph_builder.py
"""

from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np
import osmnx as ox
import pytest
import torch

from src.data import graph_builder as gb
from src.data.calibration import ARTERIAL_CRUISE_CAP_KPH
from src.data.graph_builder import (
    EDGE_FEATURE_NAMES,
    OSMID_ATTR,
    GraphBuildConfig,
    IrvineGraphData,
    build_mapping_payload,
    build_node_mapping,
    clean_graph,
    download_graph,
    extract_tensors,
    relabel_graph,
    save_artifacts,
    sorted_edges,
    validate_tensors,
)

# Deliberately non-contiguous, unsorted OSM ids: the mapping must impose order
# rather than inherit it from insertion.
OSM_IDS = [900_300, 100_100, 500_200, 700_400]

# (lon, lat) for four corners of a small block, roughly Irvine's location.
COORDS = {
    100_100: (-117.8265, 33.6846),
    500_200: (-117.8210, 33.6846),
    700_400: (-117.8210, 33.6801),
    900_300: (-117.8265, 33.6801),
}


def _make_graph(*, with_isolated: bool = False, with_dead_end: bool = False):
    """Build a synthetic OSMnx-shaped MultiDiGraph: a 4-node bidirectional ring.

    Mirrors the attributes real OSMnx output carries at this pipeline stage:
    nodes have x/y/street_count, edges have length/highway/osmid, and the graph
    carries a CRS. Speeds are intentionally absent so add_edge_speeds has work.
    """
    graph = nx.MultiDiGraph(crs="epsg:4326")

    for osmid, (lon, lat) in COORDS.items():
        graph.add_node(osmid, x=lon, y=lat, street_count=2)

    ring = [
        (100_100, 500_200, "primary", 500.0),
        (500_200, 700_400, "residential", 500.0),
        (700_400, 900_300, "primary", 500.0),
        (900_300, 100_100, "residential", 500.0),
    ]
    for u, v, highway, length in ring:
        # Both directions, so the ring is strongly connected.
        graph.add_edge(u, v, key=0, osmid=u + v, highway=highway, length=length)
        graph.add_edge(v, u, key=0, osmid=u + v, highway=highway, length=length)

    if with_isolated:
        graph.add_node(111_111, x=-117.80, y=33.70, street_count=0)

    if with_dead_end:
        # Reachable but with no way back out: a one-way routing trap that
        # strong-connectivity truncation must remove.
        graph.add_node(222_222, x=-117.8240, y=33.6820, street_count=1)
        graph.add_edge(
            100_100, 222_222, key=0, osmid=999, highway="service", length=100.0
        )

    return graph


@pytest.fixture
def raw_graph():
    return _make_graph()


@pytest.fixture
def annotated_graph(raw_graph):
    """A graph carrying speed_kph and travel_time, as produced by stage 3."""
    return gb.annotate_speeds(raw_graph, GraphBuildConfig())


@pytest.fixture
def indexed_graph(annotated_graph):
    """Relabelled to contiguous integer node indices (stage 4 output)."""
    mapping = build_node_mapping(annotated_graph)
    return relabel_graph(annotated_graph, mapping)


@pytest.fixture
def built_dir(tmp_path, indexed_graph):
    """A tmp data/processed directory with all three artifacts written."""
    tensors = extract_tensors(indexed_graph)
    node_mapping = {
        int(data[OSMID_ATTR]): idx for idx, data in indexed_graph.nodes(data=True)
    }
    payload = build_mapping_payload(
        indexed_graph,
        node_mapping,
        GraphBuildConfig(),
        num_edges=tensors["edge_index"].size(1),
    )
    save_artifacts(indexed_graph, tensors, payload, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Stage 1: download
# ---------------------------------------------------------------------------
class TestDownloadGraph:
    def test_requests_drivable_simplified_network(self, monkeypatch, raw_graph):
        """The Overpass query must ask for a drive network and simplification."""
        captured = {}

        def fake_graph_from_place(query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return raw_graph

        monkeypatch.setattr(ox, "graph_from_place", fake_graph_from_place)
        config = GraphBuildConfig(place="Irvine, California, USA")

        result = download_graph(config)

        assert result is raw_graph
        assert captured["query"] == "Irvine, California, USA"
        assert captured["network_type"] == "drive"
        assert captured["simplify"] is True
        # retain_all=True would reimport the disconnected fragments that
        # clean_graph exists to remove.
        assert captured["retain_all"] is False


# ---------------------------------------------------------------------------
# Stage 2: cleaning
# ---------------------------------------------------------------------------
class TestCleanGraph:
    def test_removes_isolated_nodes(self):
        graph = _make_graph(with_isolated=True)
        assert 111_111 in graph

        cleaned = clean_graph(graph, GraphBuildConfig())

        assert 111_111 not in cleaned
        assert cleaned.number_of_nodes() == len(COORDS)

    def test_removes_one_way_dead_end(self):
        """A node you can enter but never leave breaks routing; drop it."""
        graph = _make_graph(with_dead_end=True)

        cleaned = clean_graph(graph, GraphBuildConfig(strongly_connected=True))

        assert 222_222 not in cleaned

    def test_weak_connectivity_retains_dead_end(self):
        """Opting out of strong connectivity keeps the trap node."""
        graph = _make_graph(with_dead_end=True)

        cleaned = clean_graph(graph, GraphBuildConfig(strongly_connected=False))

        assert 222_222 in cleaned

    def test_preserves_lat_lon_coordinates(self, raw_graph):
        cleaned = clean_graph(raw_graph, GraphBuildConfig())

        for osmid, (lon, lat) in COORDS.items():
            assert cleaned.nodes[osmid]["x"] == pytest.approx(lon)
            assert cleaned.nodes[osmid]["y"] == pytest.approx(lat)


# ---------------------------------------------------------------------------
# Stage 3: speed and travel-time imputation
# ---------------------------------------------------------------------------
class TestAnnotateSpeeds:
    def test_every_edge_gets_speed_and_travel_time(self, annotated_graph):
        for _, _, data in annotated_graph.edges(data=True):
            assert data["speed_kph"] > 0
            assert data["travel_time"] > 0

    def test_uses_configured_highway_speeds(self, raw_graph):
        """Our Irvine speed table must override OSMnx's generic defaults.

        Imputation only, so the cruise cap is off: what is under test here is
        which posted limit gets attached, not what the router does with it.
        """
        config = GraphBuildConfig(
            hwy_speeds={"primary": 72.0, "residential": 40.0}, calibrate=False
        )

        annotated = gb.annotate_speeds(raw_graph, config)

        for _, _, data in annotated.edges(data=True):
            expected = 72.0 if data["highway"] == "primary" else 40.0
            assert data["speed_kph"] == pytest.approx(expected)

    def test_calibration_caps_the_posted_limit(self, raw_graph):
        """With calibration on, a 72 km/h arterial is routed at the cap."""
        config = GraphBuildConfig(hwy_speeds={"primary": 72.0, "residential": 40.0})

        annotated = gb.annotate_speeds(raw_graph, config)

        for _, _, data in annotated.edges(data=True):
            if data["highway"] == "primary":
                assert data["speed_kph"] == pytest.approx(ARTERIAL_CRUISE_CAP_KPH)
                # The limit itself is not lost, just no longer routed on.
                assert data["posted_speed_kph"] == pytest.approx(72.0)
            else:
                # Already below every cap, so untouched.
                assert data["speed_kph"] == pytest.approx(40.0)
            assert data["travel_time"] == pytest.approx(
                data["length"] / (data["speed_kph"] / 3.6)
            )

    def test_travel_time_is_length_over_speed(self, annotated_graph):
        """travel_time must be the free-flow baseline, in seconds."""
        for _, _, data in annotated_graph.edges(data=True):
            expected = data["length"] / (data["speed_kph"] * 1000 / 3600)
            assert data["travel_time"] == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Stage 4: index mapping
# ---------------------------------------------------------------------------
class TestNodeMapping:
    def test_indices_are_contiguous_zero_to_n_minus_one(self, raw_graph):
        mapping = build_node_mapping(raw_graph)

        assert sorted(mapping.values()) == list(range(len(OSM_IDS)))

    def test_mapping_is_bijective(self, raw_graph):
        mapping = build_node_mapping(raw_graph)

        assert len(set(mapping.values())) == len(mapping)
        assert set(mapping.keys()) == set(OSM_IDS)

    def test_ordering_is_sorted_by_osmid(self, raw_graph):
        """Sorted assignment is what makes repeated builds reproducible."""
        mapping = build_node_mapping(raw_graph)

        assert [osmid for osmid, _ in sorted(mapping.items(), key=lambda kv: kv[1])] == sorted(OSM_IDS)

    def test_mapping_is_deterministic_across_insertion_orders(self):
        """Same nodes added in a different order must yield the same indices."""
        forward = nx.MultiDiGraph()
        forward.add_nodes_from(OSM_IDS)
        reverse = nx.MultiDiGraph()
        reverse.add_nodes_from(reversed(OSM_IDS))

        assert build_node_mapping(forward) == build_node_mapping(reverse)

    def test_relabel_retains_original_osmid(self, annotated_graph):
        mapping = build_node_mapping(annotated_graph)

        relabelled = relabel_graph(annotated_graph, mapping)

        assert set(relabelled.nodes) == set(range(len(OSM_IDS)))
        for osmid, idx in mapping.items():
            assert relabelled.nodes[idx][OSMID_ATTR] == osmid

    def test_relabel_clears_colliding_osmid_attribute(self, annotated_graph):
        """A leftover "osmid" node attribute breaks GeoPackage export."""
        for _, data in annotated_graph.nodes(data=True):
            data["osmid"] = -1

        relabelled = relabel_graph(annotated_graph, build_node_mapping(annotated_graph))

        assert all("osmid" not in data for _, data in relabelled.nodes(data=True))

    def test_relabel_preserves_coordinates_and_edges(self, annotated_graph):
        mapping = build_node_mapping(annotated_graph)
        edge_count = annotated_graph.number_of_edges()

        relabelled = relabel_graph(annotated_graph, mapping)

        assert relabelled.number_of_edges() == edge_count
        for osmid, idx in mapping.items():
            lon, lat = COORDS[osmid]
            assert relabelled.nodes[idx]["x"] == pytest.approx(lon)
            assert relabelled.nodes[idx]["y"] == pytest.approx(lat)

    def test_edge_ordering_is_deterministic(self, indexed_graph):
        assert sorted_edges(indexed_graph) == sorted_edges(indexed_graph)
        assert sorted_edges(indexed_graph) == sorted(sorted_edges(indexed_graph))


# ---------------------------------------------------------------------------
# Stage 5: tensor extraction
# ---------------------------------------------------------------------------
class TestExtractTensors:
    def test_edge_index_shape_and_dtype(self, indexed_graph):
        tensors = extract_tensors(indexed_graph)

        assert tensors["edge_index"].shape == (2, indexed_graph.number_of_edges())
        assert tensors["edge_index"].dtype == torch.int64

    def test_edge_attr_shape_and_dtype(self, indexed_graph):
        tensors = extract_tensors(indexed_graph)

        assert tensors["edge_attr"].shape == (
            indexed_graph.number_of_edges(),
            len(EDGE_FEATURE_NAMES),
        )
        assert tensors["edge_attr"].dtype == torch.float32

    def test_edge_attr_columns_are_length_speed_traveltime(self, indexed_graph):
        """Column order is a contract with every downstream consumer."""
        tensors = extract_tensors(indexed_graph)
        assert EDGE_FEATURE_NAMES == ("length_m", "speed_kph", "travel_time_s")

        for edge_id, (u, v, key) in enumerate(sorted_edges(indexed_graph)):
            data = indexed_graph.edges[u, v, key]
            row = tensors["edge_attr"][edge_id]
            assert float(row[0]) == pytest.approx(data["length"])
            assert float(row[1]) == pytest.approx(data["speed_kph"])
            assert float(row[2]) == pytest.approx(data["travel_time"], rel=1e-5)

    def test_edge_index_columns_align_with_sorted_edges(self, indexed_graph):
        """Edge id i must address column i of edge_index and row i of attr."""
        tensors = extract_tensors(indexed_graph)

        for edge_id, (u, v, _key) in enumerate(sorted_edges(indexed_graph)):
            assert int(tensors["edge_index"][0, edge_id]) == u
            assert int(tensors["edge_index"][1, edge_id]) == v

    def test_pos_is_lat_lon_in_float64(self, indexed_graph):
        """(lat, lon) order, not (x, y) — and float64 to keep sub-metre precision."""
        tensors = extract_tensors(indexed_graph)

        assert tensors["pos"].shape == (indexed_graph.number_of_nodes(), 2)
        assert tensors["pos"].dtype == torch.float64
        for idx, data in indexed_graph.nodes(data=True):
            assert float(tensors["pos"][idx, 0]) == pytest.approx(data["y"])
            assert float(tensors["pos"][idx, 1]) == pytest.approx(data["x"])

    def test_node_indices_within_range(self, indexed_graph):
        tensors = extract_tensors(indexed_graph)

        assert int(tensors["edge_index"].min()) >= 0
        assert int(tensors["edge_index"].max()) < indexed_graph.number_of_nodes()


class TestValidateTensors:
    def _valid(self, indexed_graph):
        return extract_tensors(indexed_graph), indexed_graph.number_of_nodes()

    def test_accepts_well_formed_tensors(self, indexed_graph):
        tensors, n = self._valid(indexed_graph)

        validate_tensors(tensors, num_nodes=n)  # must not raise

    def test_rejects_wrong_edge_index_shape(self, indexed_graph):
        tensors, n = self._valid(indexed_graph)
        tensors["edge_index"] = tensors["edge_index"].T

        with pytest.raises(ValueError, match=r"edge_index must be \[2, E\]"):
            validate_tensors(tensors, num_nodes=n)

    def test_rejects_mismatched_edge_attr_rows(self, indexed_graph):
        tensors, n = self._valid(indexed_graph)
        tensors["edge_attr"] = tensors["edge_attr"][:-1]

        with pytest.raises(ValueError, match="edge_attr rows"):
            validate_tensors(tensors, num_nodes=n)

    def test_rejects_out_of_range_node_index(self, indexed_graph):
        tensors, n = self._valid(indexed_graph)
        tensors["edge_index"][0, 0] = n + 5

        with pytest.raises(ValueError, match="out-of-range"):
            validate_tensors(tensors, num_nodes=n)

    def test_rejects_nan_features(self, indexed_graph):
        tensors, n = self._valid(indexed_graph)
        tensors["edge_attr"][0, 0] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            validate_tensors(tensors, num_nodes=n)

    def test_rejects_non_positive_features(self, indexed_graph):
        """A zero speed would divide by zero downstream in the router."""
        tensors, n = self._valid(indexed_graph)
        tensors["edge_attr"][0, 1] = 0.0

        with pytest.raises(ValueError, match="non-positive"):
            validate_tensors(tensors, num_nodes=n)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestArtifacts:
    def test_writes_all_three_files(self, built_dir):
        assert (built_dir / "irvine_graph.gpkg").exists()
        assert (built_dir / "node_mapping.json").exists()
        assert (built_dir / "graph_tensors.pt").exists()

    def test_mapping_json_round_trips(self, built_dir):
        payload = json.loads((built_dir / "node_mapping.json").read_text())

        assert payload["num_nodes"] == len(OSM_IDS)
        assert payload["edge_feature_names"] == list(EDGE_FEATURE_NAMES)
        # JSON object keys are strings by spec.
        assert all(isinstance(k, str) for k in payload["osmid_to_index"])
        assert sorted(payload["osmid_to_index"].values()) == list(range(len(OSM_IDS)))

    def test_mapping_json_is_self_consistent(self, built_dir):
        payload = json.loads((built_dir / "node_mapping.json").read_text())

        for osmid_str, idx in payload["osmid_to_index"].items():
            assert payload["index_to_osmid"][idx] == int(osmid_str)
            lat, lon = payload["coordinates"][idx]
            assert (lon, lat) == pytest.approx(COORDS[int(osmid_str)])

    def test_tensor_bundle_contents(self, built_dir):
        bundle = torch.load(built_dir / "graph_tensors.pt", weights_only=True)

        assert set(bundle) >= {"edge_index", "edge_attr", "pos", "num_nodes"}
        assert bundle["edge_index"].shape[0] == 2
        assert bundle["edge_attr"].shape[1] == len(EDGE_FEATURE_NAMES)

    def test_geopackage_node_ids_match_tensor_indices(self, built_dir):
        """The GeoPackage must join to the tensors on node index."""
        geopandas = pytest.importorskip("geopandas")
        nodes = geopandas.read_file(built_dir / "irvine_graph.gpkg", layer="nodes")

        assert len(nodes) == len(OSM_IDS)
        # "osmid" is the relabelled contiguous index (OSMnx names the node
        # index that); the pre-relabel OSM id lives in OSMID_ATTR.
        assert set(nodes["osmid"].astype(int)) == set(range(len(OSM_IDS)))
        assert set(nodes[OSMID_ATTR].astype(int)) == set(OSM_IDS)


# ---------------------------------------------------------------------------
# IrvineGraphData
# ---------------------------------------------------------------------------
class TestIrvineGraphData:
    @pytest.fixture
    def graph_data(self, built_dir):
        return IrvineGraphData.load(built_dir)

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="graph_builder"):
            IrvineGraphData.load(tmp_path / "nope")

    def test_sizes(self, graph_data, indexed_graph):
        assert graph_data.num_nodes == len(OSM_IDS)
        assert graph_data.num_edges == indexed_graph.number_of_edges()
        assert len(graph_data) == len(OSM_IDS)

    def test_coordinates_returns_lat_lon(self, graph_data):
        for index in range(graph_data.num_nodes):
            osmid = graph_data.osmid(index)
            lat, lon = graph_data.coordinates(index)
            assert (lon, lat) == pytest.approx(COORDS[osmid])

    def test_latitude_longitude_accessors(self, graph_data):
        lat, lon = graph_data.coordinates(0)

        assert graph_data.latitude(0) == pytest.approx(lat)
        assert graph_data.longitude(0) == pytest.approx(lon)
        # Sanity: Irvine sits near 33.68 N, -117.82 E.
        assert 33.5 < lat < 33.8
        assert -118.0 < lon < -117.6

    def test_coordinates_many_is_vectorised(self, graph_data):
        indices = list(range(graph_data.num_nodes))

        coords = graph_data.coordinates_many(indices)

        assert coords.shape == (len(indices), 2)
        for i in indices:
            assert tuple(coords[i]) == pytest.approx(graph_data.coordinates(i))

    def test_osmid_index_round_trip(self, graph_data):
        for index in range(graph_data.num_nodes):
            assert graph_data.index_of(graph_data.osmid(index)) == index

    def test_coordinates_of_osmid(self, graph_data):
        for osmid, (lon, lat) in COORDS.items():
            assert graph_data.coordinates_of_osmid(osmid) == pytest.approx((lat, lon))

    def test_out_of_range_index_raises(self, graph_data):
        with pytest.raises(IndexError):
            graph_data.coordinates(graph_data.num_nodes)
        with pytest.raises(IndexError):
            graph_data.coordinates(-1)

    def test_unknown_osmid_raises(self, graph_data):
        with pytest.raises(KeyError, match="not in the processed graph"):
            graph_data.index_of(424_242)

    def test_nearest_node_returns_exact_match(self, graph_data):
        for index in range(graph_data.num_nodes):
            lat, lon = graph_data.coordinates(index)
            assert graph_data.nearest_node(lat, lon) == index

    def test_nearest_node_from_offset_point(self, graph_data):
        """A point nudged ~20m off a node still snaps to that node."""
        target = 0
        lat, lon = graph_data.coordinates(target)

        assert graph_data.nearest_node(lat + 0.0002, lon + 0.0002) == target

    def test_edge_endpoints_match_edge_index(self, graph_data):
        for edge_id in range(graph_data.num_edges):
            u, v = graph_data.edge_endpoints(edge_id)
            assert u == int(graph_data.edge_index[0, edge_id])
            assert v == int(graph_data.edge_index[1, edge_id])

    def test_edge_features_are_named(self, graph_data):
        features = graph_data.edge_features(0)

        assert set(features) == set(EDGE_FEATURE_NAMES)
        assert features["length_m"] > 0
        assert features["speed_kph"] > 0
        assert features["travel_time_s"] > 0

    def test_edge_features_consistent_with_speed(self, graph_data):
        """length / speed must reproduce travel_time for every edge."""
        for edge_id in range(graph_data.num_edges):
            f = graph_data.edge_features(edge_id)
            expected = f["length_m"] / (f["speed_kph"] * 1000 / 3600)
            assert f["travel_time_s"] == pytest.approx(expected, rel=1e-3)

    def test_out_of_range_edge_id_raises(self, graph_data):
        with pytest.raises(IndexError):
            graph_data.edge_endpoints(graph_data.num_edges)

    def test_metadata_carries_place(self, graph_data):
        assert graph_data.metadata["place"] == "Irvine, California, USA"

    def test_repr_is_informative(self, graph_data):
        assert "IrvineGraphData" in repr(graph_data)
        assert str(graph_data.num_nodes) in repr(graph_data)

    def test_to_pyg_data(self, graph_data):
        pytest.importorskip("torch_geometric")

        data = graph_data.to_pyg_data()

        assert data.num_nodes == graph_data.num_nodes
        assert data.edge_index.shape == (2, graph_data.num_edges)
        assert data.edge_attr.shape == (graph_data.num_edges, len(EDGE_FEATURE_NAMES))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_build_artifacts_offline(self, monkeypatch, tmp_path):
        """Full pipeline with only the Overpass call stubbed out."""
        monkeypatch.setattr(
            ox, "graph_from_place", lambda query, **kw: _make_graph(with_isolated=True)
        )
        config = GraphBuildConfig(output_dir=tmp_path)

        paths = gb.build_graph_artifacts(config)

        assert all(path.exists() for path in paths.values())
        graph_data = IrvineGraphData.load(tmp_path)
        assert graph_data.num_nodes == len(OSM_IDS)  # isolated node dropped
        assert 111_111 not in graph_data.osmid_to_index

    def test_rebuild_is_reproducible(self, monkeypatch, tmp_path):
        """Two builds over the same extract must produce identical tensors."""
        monkeypatch.setattr(ox, "graph_from_place", lambda query, **kw: _make_graph())

        first = tmp_path / "a"
        second = tmp_path / "b"
        gb.build_graph_artifacts(GraphBuildConfig(output_dir=first))
        gb.build_graph_artifacts(GraphBuildConfig(output_dir=second))

        a = IrvineGraphData.load(first)
        b = IrvineGraphData.load(second)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.edge_attr, b.edge_attr)
        assert a.index_to_osmid == b.index_to_osmid


@pytest.mark.network
class TestRealIrvineNetwork:
    """Hits the live Overpass API. Deselected by default; run with -m network."""

    def test_downloads_and_builds_real_irvine_graph(self, tmp_path):
        paths = gb.build_graph_artifacts(GraphBuildConfig(output_dir=tmp_path))
        graph_data = IrvineGraphData.load(tmp_path)

        assert all(path.exists() for path in paths.values())
        # Irvine's drivable network is on the order of 10^4 nodes; these bounds
        # are loose enough to survive OSM edits but catch a truncated download.
        assert 2_000 < graph_data.num_nodes < 100_000
        assert graph_data.num_edges > graph_data.num_nodes

        lats = graph_data.pos[:, 0]
        lons = graph_data.pos[:, 1]
        assert 33.5 < float(lats.min()) and float(lats.max()) < 33.8
        assert -118.0 < float(lons.min()) and float(lons.max()) < -117.6

        # Spot-check a known Irvine location: UC Irvine campus centre.
        index = graph_data.nearest_node(33.6405, -117.8443)
        lat, lon = graph_data.coordinates(index)
        assert math.isclose(lat, 33.6405, abs_tol=0.02)
        assert math.isclose(lon, -117.8443, abs_tol=0.02)

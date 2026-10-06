// pybind11 surface for the native routing core.
//
// GIL policy: the two methods that do real work -- find_shortest_path and the
// weight updates -- release the GIL, so a FastAPI worker keeps serving other
// requests during a search. A Graph still serialises its own callers on an
// internal mutex (it owns mutable search scratch), so parallelism comes from
// one Graph per worker thread, not from sharing one.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "router.hpp"

namespace py = pybind11;

PYBIND11_MODULE(traffic_engine, m) {
    m.doc() =
        "Native A* routing engine for irvine-traffic-ai.\n\n"
        "Road segments are edges, intersections are nodes. Edge travel times "
        "are overwritten in place from the STGNN forecaster; the topology is "
        "built once and never reallocated.";

    m.attr("__version__") = irvine::version();
    m.def("version", &irvine::version, "Engine version string.");

    m.def("haversine_meters", &irvine::haversine_meters, py::arg("lat1"),
          py::arg("lon1"), py::arg("lat2"), py::arg("lon2"),
          "Great-circle distance in metres between two lat/lon pairs.");

    py::class_<irvine::Node>(m, "Node", "An intersection.")
        .def(py::init<>())
        .def_readwrite("id", &irvine::Node::id)
        .def_readwrite("lat", &irvine::Node::lat)
        .def_readwrite("lon", &irvine::Node::lon)
        .def("__repr__", [](const irvine::Node& n) {
            return "<Node id=" + std::to_string(n.id) + " lat=" +
                   std::to_string(n.lat) + " lon=" + std::to_string(n.lon) + ">";
        });

    py::class_<irvine::Edge>(m, "Edge", "A directed road segment.")
        .def(py::init<>())
        .def_readwrite("id", &irvine::Edge::id)
        .def_readwrite("src", &irvine::Edge::src)
        .def_readwrite("dst", &irvine::Edge::dst)
        .def_readwrite("length_meters", &irvine::Edge::length_meters)
        .def_readwrite("travel_time_sec", &irvine::Edge::travel_time_sec)
        .def("__repr__", [](const irvine::Edge& e) {
            return "<Edge id=" + std::to_string(e.id) + " " +
                   std::to_string(e.src) + "->" + std::to_string(e.dst) +
                   " " + std::to_string(e.length_meters) + "m " +
                   std::to_string(e.travel_time_sec) + "s>";
        });

    py::enum_<irvine::RoadClass>(m, "RoadClass",
                                 "Functional class of a segment, as the "
                                 "intersection cost model sees it.")
        .value("FREEWAY", irvine::RoadClass::Freeway,
               "Motorway/trunk and their links: grade separated, no signals.")
        .value("ARTERIAL", irvine::RoadClass::Arterial,
               "Primary/secondary: the signalised grid.")
        .value("MINOR", irvine::RoadClass::Minor,
               "Tertiary and below: collectors, residential, service.")
        .export_values();

    m.attr("TURN_FLAG_NONE") = py::int_(irvine::turn_flag::kNone);
    m.attr("TURN_FLAG_UNPROTECTED_LEFT") =
        py::int_(irvine::turn_flag::kUnprotectedLeft);
    m.attr("TURN_FLAG_AGGRESSIVE_MERGE") =
        py::int_(irvine::turn_flag::kAggressiveMerge);

    py::class_<irvine::RoutePreferences>(
        m, "RoutePreferences",
        "What a caller will trade time for.\n\n"
        "The penalties bias the search without changing the clock: an avoided "
        "left really is a longer drive, and RouteResult.total_duration_sec "
        "reports the time that drive takes. The bias the winning path still "
        "carries comes back as RouteResult.preference_penalty_sec.")
        .def(py::init([](bool avoid_unprotected_lefts, bool low_stress_merges,
                         float unprotected_left_penalty_sec,
                         float aggressive_merge_penalty_sec) {
                 irvine::RoutePreferences p;
                 p.avoid_unprotected_lefts = avoid_unprotected_lefts;
                 p.low_stress_merges = low_stress_merges;
                 p.unprotected_left_penalty_sec = unprotected_left_penalty_sec;
                 p.aggressive_merge_penalty_sec = aggressive_merge_penalty_sec;
                 return p;
             }),
             py::arg("avoid_unprotected_lefts") = false,
             py::arg("low_stress_merges") = false,
             py::arg("unprotected_left_penalty_sec") = 45.0f,
             py::arg("aggressive_merge_penalty_sec") = 60.0f)
        .def_readwrite("avoid_unprotected_lefts",
                       &irvine::RoutePreferences::avoid_unprotected_lefts)
        .def_readwrite("low_stress_merges",
                       &irvine::RoutePreferences::low_stress_merges)
        .def_readwrite("unprotected_left_penalty_sec",
                       &irvine::RoutePreferences::unprotected_left_penalty_sec)
        .def_readwrite("aggressive_merge_penalty_sec",
                       &irvine::RoutePreferences::aggressive_merge_penalty_sec)
        .def("any", &irvine::RoutePreferences::any,
             "True when at least one avoidance is switched on.")
        .def("__repr__", [](const irvine::RoutePreferences& p) {
            if (!p.any()) {
                return std::string("<RoutePreferences fastest route>");
            }
            std::string parts;
            if (p.avoid_unprotected_lefts) {
                parts += "avoid_unprotected_lefts=+" +
                         std::to_string(static_cast<int>(
                             p.unprotected_left_penalty_sec)) + "s ";
            }
            if (p.low_stress_merges) {
                parts += "low_stress_merges=+" +
                         std::to_string(static_cast<int>(
                             p.aggressive_merge_penalty_sec)) + "s ";
            }
            parts.pop_back();
            return "<RoutePreferences " + parts + ">";
        });

    py::class_<irvine::TurnPenalties>(m, "TurnPenalties",
                                      "Seconds charged for the turn itself, on "
                                      "top of any node delay.")
        .def(py::init<>())
        .def(py::init([](float through, float right, float left, float uturn,
                         float through_band, float uturn_band) {
                 irvine::TurnPenalties p;
                 p.through_sec = through;
                 p.right_sec = right;
                 p.left_sec = left;
                 p.uturn_sec = uturn;
                 p.through_band_deg = through_band;
                 p.uturn_band_deg = uturn_band;
                 return p;
             }),
             py::arg("through_sec") = 0.0f, py::arg("right_sec") = 0.0f,
             py::arg("left_sec") = 0.0f, py::arg("uturn_sec") = 0.0f,
             py::arg("through_band_deg") = 35.0f,
             py::arg("uturn_band_deg") = 150.0f)
        .def_readwrite("through_sec", &irvine::TurnPenalties::through_sec)
        .def_readwrite("right_sec", &irvine::TurnPenalties::right_sec)
        .def_readwrite("left_sec", &irvine::TurnPenalties::left_sec)
        .def_readwrite("uturn_sec", &irvine::TurnPenalties::uturn_sec)
        .def_readwrite("through_band_deg",
                       &irvine::TurnPenalties::through_band_deg,
                       "A heading change within this many degrees is straight on.")
        .def_readwrite("uturn_band_deg", &irvine::TurnPenalties::uturn_band_deg,
                       "Beyond this it is a U-turn, not a hard left or right.")
        .def("__repr__", [](const irvine::TurnPenalties& p) {
            return "<TurnPenalties through=" + std::to_string(p.through_sec) +
                   "s right=" + std::to_string(p.right_sec) +
                   "s left=" + std::to_string(p.left_sec) +
                   "s uturn=" + std::to_string(p.uturn_sec) + "s>";
        });

    py::class_<irvine::RouteResult>(m, "RouteResult",
                                    "Outcome of a find_shortest_path call.")
        .def(py::init<>())
        .def_readwrite("found", &irvine::RouteResult::found)
        .def_readwrite("total_duration_sec",
                       &irvine::RouteResult::total_duration_sec)
        .def_readwrite("total_distance_meters",
                       &irvine::RouteResult::total_distance_meters)
        .def_readwrite("preference_penalty_sec",
                       &irvine::RouteResult::preference_penalty_sec,
                       "Search bias the winning path still carries, in "
                       "seconds. Zero when everything asked for was avoided; "
                       "never part of total_duration_sec.")
        .def_readwrite("path_node_ids", &irvine::RouteResult::path_node_ids)
        .def_readwrite("path_edge_ids", &irvine::RouteResult::path_edge_ids)
        .def_readwrite("nodes_expanded", &irvine::RouteResult::nodes_expanded)
        .def("__bool__", [](const irvine::RouteResult& r) { return r.found; })
        .def("__repr__", [](const irvine::RouteResult& r) {
            if (!r.found) {
                return std::string("<RouteResult found=False>");
            }
            return "<RouteResult " + std::to_string(r.path_node_ids.size()) +
                   " nodes, " + std::to_string(r.total_duration_sec) + "s, " +
                   std::to_string(r.total_distance_meters) + "m>";
        });

    py::class_<irvine::Graph>(m, "Graph",
                              "Contiguous road graph with a CSR adjacency.")
        .def(py::init<std::size_t>(), py::arg("num_nodes"),
             "Allocate storage for num_nodes intersections.")

        .def("add_node", &irvine::Graph::add_node, py::arg("id"),
             py::arg("lat"), py::arg("lon"),
             "Place an intersection at index `id`, which must be in "
             "[0, num_nodes).")

        .def("add_edge", &irvine::Graph::add_edge, py::arg("edge_id"),
             py::arg("u"), py::arg("v"), py::arg("length_meters"),
             py::arg("travel_time_sec"),
             "Add a directed segment u -> v at index `edge_id`. Edge ids must "
             "be dense over [0, num_edges).")

        .def("finalize", &irvine::Graph::finalize,
             py::call_guard<py::gil_scoped_release>(),
             "Build the CSR adjacency. Happens automatically on first use; "
             "call it explicitly to keep build cost out of a benchmark.")

        .def("update_edge_weights", &irvine::Graph::update_edge_weights,
             py::arg("predicted_times"),
             py::call_guard<py::gil_scoped_release>(),
             "Overwrite every edge travel time, in edge-id order. O(E), no "
             "reallocation, topology untouched.")

        .def("update_edge_weights_subset",
             &irvine::Graph::update_edge_weights_subset, py::arg("edge_ids"),
             py::arg("predicted_times"),
             py::call_guard<py::gil_scoped_release>(),
             "Overwrite travel times for the listed edge ids only. O(k). Use "
             "this when the forecaster covers a subset of the network.")

        .def("set_node_delays", &irvine::Graph::set_node_delays,
             py::arg("priority_sec"), py::arg("yield_sec") = std::vector<float>(),
             py::call_guard<py::gil_scoped_release>(),
             "Per-node intersection delay in seconds. `priority_sec` is charged "
             "to every movement through the node (a signal); `yield_sec` only "
             "to movements entering or leaving a MINOR-class edge (a stop "
             "sign). Empty vectors clear both.")

        .def("set_edge_road_classes", &irvine::Graph::set_edge_road_classes,
             py::arg("classes"), py::call_guard<py::gil_scoped_release>(),
             "RoadClass per edge id. Read only by the yield rule.")

        .def("set_turn_penalties", &irvine::Graph::set_turn_penalties,
             py::arg("penalties"), py::call_guard<py::gil_scoped_release>(),
             "Seconds charged for a left, right, through or U-turn movement.")

        .def_property_readonly("turn_penalties", &irvine::Graph::turn_penalties)

        .def("set_edge_bearings", &irvine::Graph::set_edge_bearings,
             py::arg("entry_deg"), py::arg("exit_deg"),
             py::call_guard<py::gil_scoped_release>(),
             "Compass bearing at each edge's start and end, for turn angles. "
             "Defaults to the straight src -> dst chord, which is exact only "
             "for a straight segment.")

        .def("transition_cost", &irvine::Graph::transition_cost,
             py::arg("in_edge"), py::arg("out_edge"),
             py::arg("preferences") = irvine::RoutePreferences{},
             py::call_guard<py::gil_scoped_release>(),
             "Seconds added at the node two consecutive edges share: node "
             "delays plus the turn penalty.")

        .def("set_turn_flags", &irvine::Graph::set_turn_flags,
             py::arg("in_edge_ids"), py::arg("out_edge_ids"), py::arg("flags"),
             py::call_guard<py::gil_scoped_release>(),
             "Per-movement hazard flags as a sparse (in_edge, out_edge) -> "
             "flags table -- the shape graph_builder's TurnTable exports. "
             "Finalizes the graph, so set edge bearings first.")

        .def("turn_flags", &irvine::Graph::turn_flags, py::arg("in_edge"),
             py::arg("out_edge"), py::call_guard<py::gil_scoped_release>(),
             "Flag bits for one movement; 0 when nothing is set.")

        .def_property_readonly("has_turn_flags", &irvine::Graph::has_turn_flags)

        .def("score_path", &irvine::Graph::score_path, py::arg("edge_ids"),
             py::arg("preferences") = irvine::RoutePreferences{},
             py::call_guard<py::gil_scoped_release>(),
             "Total seconds for a path of consecutive edge ids under this "
             "graph's weights: travel time plus every interior transition "
             "cost. The quantity find_shortest_path minimises.")

        .def("preference_penalty", &irvine::Graph::preference_penalty,
             py::arg("edge_ids"), py::arg("preferences"),
             py::call_guard<py::gil_scoped_release>(),
             "Seconds of preference bias a path carries: the part of its "
             "search cost that is not time on the road.")

        .def_property_readonly("has_intersection_costs",
                               &irvine::Graph::has_intersection_costs,
                               "True once any node delay or turn penalty is set.")

        .def("find_shortest_path", &irvine::Graph::find_shortest_path,
             py::arg("start_node"), py::arg("end_node"),
             py::arg("preferences") = irvine::RoutePreferences{},
             py::call_guard<py::gil_scoped_release>(),
             "Minimum-cost route via A* with a Haversine lower bound. Cost is "
             "edge travel time plus intersection delays and turn penalties, "
             "plus any penalty `preferences` attaches to a flagged movement. "
             "`nodes_expanded` counts expanded search states, which are "
             "directed edges rather than nodes.")

        .def("node", &irvine::Graph::node, py::arg("id"),
             py::return_value_policy::reference_internal)
        .def("edge", &irvine::Graph::edge, py::arg("id"),
             py::return_value_policy::reference_internal)

        .def_property_readonly("num_nodes", &irvine::Graph::num_nodes)
        .def_property_readonly("num_edges", &irvine::Graph::num_edges)
        .def_property_readonly("is_finalized", &irvine::Graph::is_finalized)
        .def_property_readonly(
            "max_speed_mps", &irvine::Graph::max_speed_mps,
            "Fastest edge in the graph. The A* heuristic divides Haversine "
            "metres by this to get an admissible lower bound in seconds.")

        .def("__len__", &irvine::Graph::num_nodes)
        .def("__repr__", [](const irvine::Graph& g) {
            return "<Graph nodes=" + std::to_string(g.num_nodes()) +
                   " edges=" + std::to_string(g.num_edges()) +
                   " finalized=" + (g.is_finalized() ? "True" : "False") + ">";
        });
}

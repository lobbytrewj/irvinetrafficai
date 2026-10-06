#include "router.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace irvine {
namespace {

// Initial open-set capacity. A* over a city-scale road graph expands a few
// thousand nodes; reserving up front keeps the heap from reallocating mid
// query, which is the only allocation the search would otherwise do.
constexpr std::size_t kOpenSetReserve = 4096;

[[noreturn]] void throw_out_of_range(const char* what, int id, std::size_t size) {
    throw std::out_of_range(std::string(what) + " id " + std::to_string(id) +
                            " out of range [0, " + std::to_string(size) + ")");
}

constexpr double kPi = 3.14159265358979323846;

double to_radians(double degrees) noexcept { return degrees * (kPi / 180.0); }

// Initial compass bearing from one point to another, in degrees clockwise
// from north and normalised to [0, 360).
double bearing_degrees(double lat1, double lon1, double lat2,
                       double lon2) noexcept {
    const double phi1 = to_radians(lat1);
    const double phi2 = to_radians(lat2);
    const double dlambda = to_radians(lon2 - lon1);

    const double y = std::sin(dlambda) * std::cos(phi2);
    const double x = std::cos(phi1) * std::sin(phi2) -
                     std::sin(phi1) * std::cos(phi2) * std::cos(dlambda);
    const double degrees = std::atan2(y, x) * (180.0 / kPi);
    return degrees < 0.0 ? degrees + 360.0 : degrees;
}

// Signed heading change in (-180, 180]. Positive is clockwise, which in
// right-hand traffic is a right turn.
float signed_turn_degrees(float from_deg, float to_deg) noexcept {
    float delta = std::fmod(to_deg - from_deg + 540.0f, 360.0f) - 180.0f;
    // fmod keeps the sign of the numerator, which the +540 above rules out,
    // so delta is already in [-180, 180).
    return delta;
}

}  // namespace

const char* version() noexcept { return "0.4.0"; }

double haversine_meters(double lat1, double lon1, double lat2,
                        double lon2) noexcept {
    const double phi1 = to_radians(lat1);
    const double phi2 = to_radians(lat2);
    const double dphi = phi2 - phi1;
    const double dlambda = to_radians(lon2 - lon1);

    const double sin_dphi = std::sin(dphi * 0.5);
    const double sin_dlambda = std::sin(dlambda * 0.5);
    const double a = sin_dphi * sin_dphi +
                     std::cos(phi1) * std::cos(phi2) * sin_dlambda * sin_dlambda;
    // asin(sqrt(a)) rather than atan2: `a` is clamped below 1 by construction
    // up to rounding, and this form is the cheaper of the two.
    return 2.0 * kEarthRadiusMeters * std::asin(std::sqrt(std::min(1.0, a)));
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
Graph::Graph(std::size_t num_nodes)
    : nodes_(num_nodes), node_assigned_(num_nodes, false) {}

void Graph::add_node(int id, double lat, double lon) {
    if (id < 0 || static_cast<std::size_t>(id) >= nodes_.size()) {
        throw_out_of_range("node", id, nodes_.size());
    }
    nodes_[static_cast<std::size_t>(id)] = Node{id, lat, lon};
    node_assigned_[static_cast<std::size_t>(id)] = true;
    // Adding a node after finalize() would leave the CSR stale, and the turn
    // flag table is indexed off it.
    finalized_ = false;
    clear_turn_flags();
}

void Graph::add_edge(int edge_id, int u, int v, float length_meters,
                     float travel_time_sec) {
    if (edge_id < 0) {
        throw std::out_of_range("edge id must be non-negative, got " +
                                std::to_string(edge_id));
    }
    if (!std::isfinite(length_meters) || length_meters < 0.0f) {
        throw std::invalid_argument("edge " + std::to_string(edge_id) +
                                    " has non-finite or negative length");
    }
    // A* and Dijkstra both assume non-negative arc costs; a negative travel
    // time would silently produce a wrong path rather than an error later.
    if (!std::isfinite(travel_time_sec) || travel_time_sec < 0.0f) {
        throw std::invalid_argument("edge " + std::to_string(edge_id) +
                                    " has non-finite or negative travel time");
    }

    const auto index = static_cast<std::size_t>(edge_id);
    if (index >= edges_.size()) {
        edges_.resize(index + 1);
        edge_assigned_.resize(index + 1, false);
    }
    edges_[index] = Edge{edge_id, u, v, length_meters, travel_time_sec};
    edge_assigned_[index] = true;
    finalized_ = false;
    clear_turn_flags();
}

void Graph::finalize() {
    if (finalized_) {
        return;
    }
    const std::size_t num_nodes = nodes_.size();
    const std::size_t num_edges = edges_.size();

    for (std::size_t i = 0; i < num_nodes; ++i) {
        if (!node_assigned_[i]) {
            throw std::invalid_argument(
                "node " + std::to_string(i) +
                " was never added; every node needs a lat/lon for the heuristic");
        }
    }
    for (std::size_t i = 0; i < num_edges; ++i) {
        if (!edge_assigned_[i]) {
            throw std::invalid_argument("edge ids must be dense over [0, " +
                                        std::to_string(num_edges) +
                                        "); id " + std::to_string(i) +
                                        " was never added");
        }
        const Edge& e = edges_[i];
        if (e.src < 0 || static_cast<std::size_t>(e.src) >= num_nodes) {
            throw_out_of_range("edge source node", e.src, num_nodes);
        }
        if (e.dst < 0 || static_cast<std::size_t>(e.dst) >= num_nodes) {
            throw_out_of_range("edge target node", e.dst, num_nodes);
        }
    }

    // Counting sort into CSR order: one pass to count out-degrees, a prefix
    // sum for the offsets, one pass to place. O(N + E), no per-node vectors.
    adj_offsets_.assign(num_nodes + 1, 0);
    for (const Edge& e : edges_) {
        ++adj_offsets_[static_cast<std::size_t>(e.src) + 1];
    }
    for (std::size_t n = 0; n < num_nodes; ++n) {
        adj_offsets_[n + 1] += adj_offsets_[n];
    }

    adjacency_.resize(num_edges);
    edge_to_slot_.assign(num_edges, -1);
    std::vector<std::int32_t> cursor(adj_offsets_.begin(), adj_offsets_.end() - 1);
    for (const Edge& e : edges_) {
        const auto slot = static_cast<std::size_t>(cursor[static_cast<std::size_t>(e.src)]++);
        adjacency_[slot] = AdjEntry{e.dst, e.id, e.travel_time_sec};
        edge_to_slot_[static_cast<std::size_t>(e.id)] = static_cast<std::int32_t>(slot);
    }

    // Scratch is per *state*, and a state is a directed edge.
    g_score_.assign(num_edges, 0.0f);
    h_cache_.assign(num_edges, 0.0f);
    came_from_state_.assign(num_edges, -1);
    visit_stamp_.assign(num_edges, 0);
    generation_ = 0;

    compute_chord_bearings();
    refresh_max_speed();
    refresh_intersection_costs();
    finalized_ = true;
    // Re-pack any flags supplied before the topology settled.
    build_turn_flag_index();
}

void Graph::compute_chord_bearings() {
    // Only fills what set_edge_bearings has not already supplied. The chord
    // bearing is exact for a straight segment; on a curving arterial the true
    // heading at the far end can differ by tens of degrees, which is why the
    // setter exists for callers that hold the real geometry.
    if (edge_entry_bearing_.size() == edges_.size() &&
        edge_exit_bearing_.size() == edges_.size()) {
        return;
    }
    edge_entry_bearing_.assign(edges_.size(), 0.0f);
    edge_exit_bearing_.assign(edges_.size(), 0.0f);
    for (const Edge& e : edges_) {
        const Node& u = nodes_[static_cast<std::size_t>(e.src)];
        const Node& v = nodes_[static_cast<std::size_t>(e.dst)];
        const auto chord =
            static_cast<float>(bearing_degrees(u.lat, u.lon, v.lat, v.lon));
        edge_entry_bearing_[static_cast<std::size_t>(e.id)] = chord;
        edge_exit_bearing_[static_cast<std::size_t>(e.id)] = chord;
    }
}

void Graph::ensure_finalized() {
    if (!finalized_) {
        finalize();
    }
}

const Node& Graph::node(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= nodes_.size()) {
        throw_out_of_range("node", id, nodes_.size());
    }
    return nodes_[static_cast<std::size_t>(id)];
}

const Edge& Graph::edge(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= edges_.size()) {
        throw_out_of_range("edge", id, edges_.size());
    }
    return edges_[static_cast<std::size_t>(id)];
}

// ---------------------------------------------------------------------------
// Dynamic weights
// ---------------------------------------------------------------------------
void Graph::write_travel_time(int edge_id, float seconds) noexcept {
    const auto index = static_cast<std::size_t>(edge_id);
    edges_[index].travel_time_sec = seconds;
    const std::int32_t slot = edge_to_slot_[index];
    if (slot >= 0) {
        adjacency_[static_cast<std::size_t>(slot)].travel_time_sec = seconds;
    }
}

void Graph::refresh_max_speed() noexcept {
    // The Haversine heuristic returns metres; A* costs are seconds. Dividing
    // by the fastest edge in the graph converts one to the other while keeping
    // the bound admissible: no route can cover d metres in less than
    // d / max_speed seconds. Recomputed on every weight update, because a
    // stale (too high) speed is still admissible but weaker, while a stale
    // too-low one would overestimate and could return a suboptimal path.
    double fastest = 0.0;
    for (const Edge& e : edges_) {
        if (e.travel_time_sec > 0.0f && e.length_meters > 0.0f) {
            fastest = std::max(fastest, static_cast<double>(e.length_meters) /
                                            static_cast<double>(e.travel_time_sec));
        }
    }
    max_speed_mps_ = fastest;
}

void Graph::update_edge_weights(const std::vector<float>& new_travel_times) {
    std::lock_guard<std::mutex> guard(search_mutex_);
    ensure_finalized();
    if (new_travel_times.size() != edges_.size()) {
        throw std::invalid_argument(
            "expected " + std::to_string(edges_.size()) + " travel times, got " +
            std::to_string(new_travel_times.size()));
    }

    const std::size_t count = edges_.size();
    for (std::size_t i = 0; i < count; ++i) {
        const float seconds = new_travel_times[i];
        if (!std::isfinite(seconds) || seconds < 0.0f) {
            throw std::invalid_argument("travel time for edge " +
                                        std::to_string(i) +
                                        " is non-finite or negative");
        }
        write_travel_time(static_cast<int>(i), seconds);
    }
    refresh_max_speed();
}

void Graph::update_edge_weights_subset(const std::vector<int>& edge_ids,
                                       const std::vector<float>& new_travel_times) {
    std::lock_guard<std::mutex> guard(search_mutex_);
    ensure_finalized();
    if (edge_ids.size() != new_travel_times.size()) {
        throw std::invalid_argument(
            "edge_ids and new_travel_times must be the same length, got " +
            std::to_string(edge_ids.size()) + " and " +
            std::to_string(new_travel_times.size()));
    }

    for (std::size_t i = 0; i < edge_ids.size(); ++i) {
        const int edge_id = edge_ids[i];
        if (edge_id < 0 || static_cast<std::size_t>(edge_id) >= edges_.size()) {
            throw_out_of_range("edge", edge_id, edges_.size());
        }
        const float seconds = new_travel_times[i];
        if (!std::isfinite(seconds) || seconds < 0.0f) {
            throw std::invalid_argument("travel time for edge " +
                                        std::to_string(edge_id) +
                                        " is non-finite or negative");
        }
        write_travel_time(edge_id, seconds);
    }
    refresh_max_speed();
}

// ---------------------------------------------------------------------------
// Intersection costs
// ---------------------------------------------------------------------------
void Graph::set_node_delays(const std::vector<float>& priority_sec,
                            const std::vector<float>& yield_sec) {
    std::lock_guard<std::mutex> guard(search_mutex_);

    if (priority_sec.empty() && yield_sec.empty()) {
        node_priority_delay_.clear();
        node_yield_delay_.clear();
        refresh_intersection_costs();
        return;
    }
    for (const auto* vec : {&priority_sec, &yield_sec}) {
        if (!vec->empty() && vec->size() != nodes_.size()) {
            throw std::invalid_argument(
                "node delays must have one entry per node (" +
                std::to_string(nodes_.size()) + "), got " +
                std::to_string(vec->size()));
        }
        for (const float seconds : *vec) {
            // A negative delay would make an intersection worth driving
            // through twice, and A* assumes it cannot be.
            if (!std::isfinite(seconds) || seconds < 0.0f) {
                throw std::invalid_argument(
                    "node delay is non-finite or negative");
            }
        }
    }

    node_priority_delay_ = priority_sec;
    node_yield_delay_ = yield_sec;
    node_priority_delay_.resize(nodes_.size(), 0.0f);
    node_yield_delay_.resize(nodes_.size(), 0.0f);
    refresh_intersection_costs();
}

void Graph::set_edge_road_classes(const std::vector<std::uint8_t>& classes) {
    std::lock_guard<std::mutex> guard(search_mutex_);
    if (!classes.empty() && classes.size() != edges_.size()) {
        throw std::invalid_argument(
            "edge road classes must have one entry per edge (" +
            std::to_string(edges_.size()) + "), got " +
            std::to_string(classes.size()));
    }
    for (const std::uint8_t value : classes) {
        if (value > static_cast<std::uint8_t>(RoadClass::Minor)) {
            throw std::invalid_argument("unknown road class " +
                                        std::to_string(value));
        }
    }
    edge_road_class_ = classes;
}

void Graph::set_turn_penalties(const TurnPenalties& penalties) {
    std::lock_guard<std::mutex> guard(search_mutex_);
    for (const float seconds : {penalties.through_sec, penalties.right_sec,
                                penalties.left_sec, penalties.uturn_sec}) {
        if (!std::isfinite(seconds) || seconds < 0.0f) {
            throw std::invalid_argument(
                "turn penalties must be finite and non-negative");
        }
    }
    if (!(penalties.through_band_deg >= 0.0f &&
          penalties.through_band_deg <= penalties.uturn_band_deg &&
          penalties.uturn_band_deg <= 180.0f)) {
        throw std::invalid_argument(
            "turn bands must satisfy 0 <= through <= uturn <= 180 degrees");
    }
    turns_ = penalties;
    refresh_intersection_costs();
}

void Graph::set_edge_bearings(const std::vector<float>& entry_deg,
                              const std::vector<float>& exit_deg) {
    std::lock_guard<std::mutex> guard(search_mutex_);
    if (entry_deg.size() != edges_.size() || exit_deg.size() != edges_.size()) {
        throw std::invalid_argument(
            "edge bearings must have one entry per edge (" +
            std::to_string(edges_.size()) + "), got " +
            std::to_string(entry_deg.size()) + " and " +
            std::to_string(exit_deg.size()));
    }
    for (std::size_t i = 0; i < entry_deg.size(); ++i) {
        if (!std::isfinite(entry_deg[i]) || !std::isfinite(exit_deg[i])) {
            throw std::invalid_argument("edge " + std::to_string(i) +
                                        " has a non-finite bearing");
        }
    }
    edge_entry_bearing_ = entry_deg;
    edge_exit_bearing_ = exit_deg;
}

void Graph::refresh_intersection_costs() noexcept {
    const auto any_positive = [](const std::vector<float>& values) {
        for (const float value : values) {
            if (value > 0.0f) {
                return true;
            }
        }
        return false;
    };
    intersection_costs_ = any_positive(node_priority_delay_) ||
                          any_positive(node_yield_delay_) ||
                          turns_.through_sec > 0.0f || turns_.right_sec > 0.0f ||
                          turns_.left_sec > 0.0f || turns_.uturn_sec > 0.0f;
}

float Graph::turn_penalty(int in_edge, int out_edge) const noexcept {
    if (edge_entry_bearing_.empty()) {
        return 0.0f;
    }
    const float delta = signed_turn_degrees(
        edge_exit_bearing_[static_cast<std::size_t>(in_edge)],
        edge_entry_bearing_[static_cast<std::size_t>(out_edge)]);
    const float magnitude = std::fabs(delta);

    if (magnitude <= turns_.through_band_deg) {
        return turns_.through_sec;
    }
    if (magnitude >= turns_.uturn_band_deg) {
        return turns_.uturn_sec;
    }
    // Bearings grow clockwise, so a positive change is a right turn.
    return delta > 0.0f ? turns_.right_sec : turns_.left_sec;
}

float Graph::transition_cost_unchecked(int in_edge, int out_edge) const noexcept {
    if (!intersection_costs_) {
        return 0.0f;
    }
    const auto node = static_cast<std::size_t>(
        edges_[static_cast<std::size_t>(in_edge)].dst);

    float delay = 0.0f;
    if (!node_priority_delay_.empty()) {
        delay += node_priority_delay_[node];
    }
    if (!node_yield_delay_.empty() && node_yield_delay_[node] > 0.0f &&
        !edge_road_class_.empty()) {
        // A stop sign on a residential street costs the driver who is on --
        // or turning onto -- that street, and nobody else.
        const auto minor = static_cast<std::uint8_t>(RoadClass::Minor);
        if (edge_road_class_[static_cast<std::size_t>(in_edge)] == minor ||
            edge_road_class_[static_cast<std::size_t>(out_edge)] == minor) {
            delay += node_yield_delay_[node];
        }
    }
    return delay + turn_penalty(in_edge, out_edge);
}

double Graph::transition_cost(int in_edge, int out_edge,
                             const RoutePreferences& preferences) const {
    if (in_edge < 0 || static_cast<std::size_t>(in_edge) >= edges_.size()) {
        throw_out_of_range("edge", in_edge, edges_.size());
    }
    if (out_edge < 0 || static_cast<std::size_t>(out_edge) >= edges_.size()) {
        throw_out_of_range("edge", out_edge, edges_.size());
    }
    if (edges_[static_cast<std::size_t>(in_edge)].dst !=
        edges_[static_cast<std::size_t>(out_edge)].src) {
        throw std::invalid_argument(
            "edges " + std::to_string(in_edge) + " and " +
            std::to_string(out_edge) +
            " do not meet: the first ends at a different node than the second "
            "begins at");
    }
    double seconds = transition_cost_unchecked(in_edge, out_edge);
    const PenaltyTable table = penalty_table(preferences);
    if (table.active && !turn_flags_.empty()) {
        seconds += table.by_flags[turn_flags(in_edge, out_edge) & turn_flag::kAll];
    }
    return seconds;
}

double Graph::score_path(const std::vector<int>& edge_ids,
                         const RoutePreferences& preferences) const {
    double total = 0.0;
    for (std::size_t i = 0; i < edge_ids.size(); ++i) {
        const int edge_id = edge_ids[i];
        if (edge_id < 0 || static_cast<std::size_t>(edge_id) >= edges_.size()) {
            throw_out_of_range("edge", edge_id, edges_.size());
        }
        total += edges_[static_cast<std::size_t>(edge_id)].travel_time_sec;
        if (i > 0) {
            // Also validates that consecutive edges actually connect, so a
            // path assembled by mistake is rejected rather than scored.
            total += transition_cost(edge_ids[i - 1], edge_id, preferences);
        }
    }
    return total;
}

double Graph::preference_penalty(const std::vector<int>& edge_ids,
                                 const RoutePreferences& preferences) const {
    const PenaltyTable table = penalty_table(preferences);
    if (!table.active || turn_flags_.empty() || edge_ids.size() < 2) {
        return 0.0;
    }

    double total = 0.0;
    for (std::size_t i = 1; i < edge_ids.size(); ++i) {
        total += table.by_flags[turn_flags(edge_ids[i - 1], edge_ids[i]) &
                                turn_flag::kAll];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Movement flags and preferences
// ---------------------------------------------------------------------------
void Graph::clear_turn_flags() noexcept {
    turn_offsets_.clear();
    turn_flags_.clear();
}

std::int32_t Graph::adjacency_slot(int out_edge) const noexcept {
    return edge_to_slot_[static_cast<std::size_t>(out_edge)];
}

void Graph::set_turn_flags(const std::vector<int>& in_edge_ids,
                           const std::vector<int>& out_edge_ids,
                           const std::vector<std::uint8_t>& flags) {
    if (in_edge_ids.size() != out_edge_ids.size() ||
        in_edge_ids.size() != flags.size()) {
        throw std::invalid_argument(
            "in_edge_ids, out_edge_ids and flags must be the same length, got " +
            std::to_string(in_edge_ids.size()) + ", " +
            std::to_string(out_edge_ids.size()) + " and " +
            std::to_string(flags.size()));
    }

    // One acquisition for the whole thing: finalize() deliberately does not
    // lock, because every caller of it already holds this mutex.
    std::lock_guard<std::mutex> guard(search_mutex_);

    for (std::size_t i = 0; i < in_edge_ids.size(); ++i) {
        for (const int edge_id : {in_edge_ids[i], out_edge_ids[i]}) {
            if (edge_id < 0 ||
                static_cast<std::size_t>(edge_id) >= edges_.size()) {
                throw_out_of_range("edge", edge_id, edges_.size());
            }
        }
        if ((flags[i] & ~turn_flag::kAll) != 0u) {
            throw std::invalid_argument("unknown turn flag bits in " +
                                        std::to_string(flags[i]));
        }
    }

    pending_turn_in_ = in_edge_ids;
    pending_turn_out_ = out_edge_ids;
    pending_turn_flags_ = flags;
    clear_turn_flags();

    if (finalized_) {
        build_turn_flag_index();
    } else {
        // finalize() re-packs on its way out.
        ensure_finalized();
    }
}

void Graph::build_turn_flag_index() {
    if (pending_turn_in_.empty()) {
        clear_turn_flags();
        return;
    }

    // One slot per (in_edge, out_edge of in_edge's destination) pair, laid out
    // so that everything reachable from one edge is contiguous. The count for
    // edge e is the out-degree of its destination, which the CSR already knows.
    const std::size_t num_edges = edges_.size();
    turn_offsets_.assign(num_edges + 1, 0);
    for (std::size_t e = 0; e < num_edges; ++e) {
        const auto dst = static_cast<std::size_t>(edges_[e].dst);
        turn_offsets_[e + 1] = adj_offsets_[dst + 1] - adj_offsets_[dst];
    }
    for (std::size_t e = 0; e < num_edges; ++e) {
        turn_offsets_[e + 1] += turn_offsets_[e];
    }
    turn_flags_.assign(static_cast<std::size_t>(turn_offsets_[num_edges]),
                       turn_flag::kNone);

    for (std::size_t i = 0; i < pending_turn_in_.size(); ++i) {
        const int in_edge = pending_turn_in_[i];
        const int out_edge = pending_turn_out_[i];
        const Edge& incoming = edges_[static_cast<std::size_t>(in_edge)];
        const Edge& outgoing = edges_[static_cast<std::size_t>(out_edge)];
        if (incoming.dst != outgoing.src) {
            throw std::invalid_argument(
                "turn flag for edges " + std::to_string(in_edge) + " and " +
                std::to_string(out_edge) +
                " which do not meet: the first ends at a different node than "
                "the second begins at");
        }
        const auto node = static_cast<std::size_t>(outgoing.src);
        const std::int32_t slot = adjacency_slot(out_edge);
        const std::int32_t within = slot - adj_offsets_[node];
        turn_flags_[static_cast<std::size_t>(
            turn_offsets_[static_cast<std::size_t>(in_edge)] + within)] |=
            pending_turn_flags_[i];
    }
}

std::uint8_t Graph::turn_flags(int in_edge, int out_edge) const {
    if (in_edge < 0 || static_cast<std::size_t>(in_edge) >= edges_.size()) {
        throw_out_of_range("edge", in_edge, edges_.size());
    }
    if (out_edge < 0 || static_cast<std::size_t>(out_edge) >= edges_.size()) {
        throw_out_of_range("edge", out_edge, edges_.size());
    }
    const Edge& incoming = edges_[static_cast<std::size_t>(in_edge)];
    const Edge& outgoing = edges_[static_cast<std::size_t>(out_edge)];
    if (incoming.dst != outgoing.src) {
        throw std::invalid_argument(
            "edges " + std::to_string(in_edge) + " and " +
            std::to_string(out_edge) + " do not meet");
    }
    if (turn_flags_.empty()) {
        return turn_flag::kNone;
    }
    const auto node = static_cast<std::size_t>(outgoing.src);
    const std::int32_t within = adjacency_slot(out_edge) - adj_offsets_[node];
    return turn_flags_[static_cast<std::size_t>(
        turn_offsets_[static_cast<std::size_t>(in_edge)] + within)];
}

Graph::PenaltyTable Graph::penalty_table(
    const RoutePreferences& preferences) noexcept {
    PenaltyTable table;
    table.active = preferences.any();
    if (!table.active) {
        return table;
    }
    const float left = preferences.avoid_unprotected_lefts
                           ? std::max(0.0f, preferences.unprotected_left_penalty_sec)
                           : 0.0f;
    const float merge = preferences.low_stress_merges
                            ? std::max(0.0f, preferences.aggressive_merge_penalty_sec)
                            : 0.0f;
    // Indexed by the flag bits themselves, so the inner loop is one load and
    // no branches.
    table.by_flags[turn_flag::kNone] = 0.0f;
    table.by_flags[turn_flag::kUnprotectedLeft] = left;
    table.by_flags[turn_flag::kAggressiveMerge] = merge;
    table.by_flags[turn_flag::kAll] = left + merge;
    return table;
}

// ---------------------------------------------------------------------------
// A*
// ---------------------------------------------------------------------------
void Graph::begin_search() {
    // Stamps mark which scratch entries belong to the current query, so state
    // is invalidated by one increment rather than by clearing N entries. On
    // wrap-around the stamps really do have to be cleared -- once every 4
    // billion queries.
    if (++generation_ == 0) {
        std::fill(visit_stamp_.begin(), visit_stamp_.end(), 0u);
        generation_ = 1;
    }
}

float Graph::heuristic(int node, double goal_lat, double goal_lon) noexcept {
    if (max_speed_mps_ <= 0.0) {
        return 0.0f;  // No usable speed: degenerate to Dijkstra, still correct.
    }
    const Node& n = nodes_[static_cast<std::size_t>(node)];
    const double metres = haversine_meters(n.lat, n.lon, goal_lat, goal_lon);
    return static_cast<float>(metres / max_speed_mps_);
}

RouteResult Graph::reconstruct(int start_node, int goal_edge,
                               std::uint64_t expanded,
                               const RoutePreferences& preferences) const {
    RouteResult result;
    result.found = true;
    result.nodes_expanded = expanded;

    // Walk predecessor *states* back from the edge that reached the goal, then
    // reverse. Totals are summed in double from the canonical edge records:
    // the search accumulates in float for cache density, which is plenty for
    // path selection but not what we want to hand back as a duration.
    for (std::int32_t edge_id = goal_edge; edge_id != -1;
         edge_id = came_from_state_[static_cast<std::size_t>(edge_id)]) {
        result.path_edge_ids.push_back(edge_id);
    }
    std::reverse(result.path_edge_ids.begin(), result.path_edge_ids.end());

    result.path_node_ids.reserve(result.path_edge_ids.size() + 1);
    result.path_node_ids.push_back(start_node);
    for (std::size_t i = 0; i < result.path_edge_ids.size(); ++i) {
        const Edge& e = edges_[static_cast<std::size_t>(result.path_edge_ids[i])];
        result.path_node_ids.push_back(e.dst);
        result.total_duration_sec += e.travel_time_sec;
        result.total_distance_meters += e.length_meters;
        if (i > 0) {
            // The same intersection cost the search paid to get here.
            result.total_duration_sec += transition_cost_unchecked(
                result.path_edge_ids[i - 1], result.path_edge_ids[i]);
        }
    }
    // Reported apart from the duration, never inside it. The penalty is what
    // the driver was willing to pay to avoid something, not time they spend.
    result.preference_penalty_sec =
        preference_penalty(result.path_edge_ids, preferences);
    return result;
}

RouteResult Graph::find_shortest_path(int start_node, int end_node,
                                      const RoutePreferences& preferences) {
    if (start_node < 0 || static_cast<std::size_t>(start_node) >= nodes_.size()) {
        throw_out_of_range("start node", start_node, nodes_.size());
    }
    if (end_node < 0 || static_cast<std::size_t>(end_node) >= nodes_.size()) {
        throw_out_of_range("end node", end_node, nodes_.size());
    }

    std::lock_guard<std::mutex> guard(search_mutex_);
    ensure_finalized();

    RouteResult result;
    if (start_node == end_node) {
        result.found = true;
        result.path_node_ids.push_back(start_node);
        return result;
    }

    begin_search();
    const Node& goal = nodes_[static_cast<std::size_t>(end_node)];

    // Built once per query, then indexed by the flag bits directly: the inner
    // loop pays one byte load and one float load, with no branch on which
    // preference is switched on.
    const PenaltyTable penalties = penalty_table(preferences);
    const bool weigh_flags = penalties.active && !turn_flags_.empty();

    std::vector<QueueEntry> heap_storage;
    heap_storage.reserve(kOpenSetReserve);
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>>
        open(std::greater<QueueEntry>{}, std::move(heap_storage));

    // Seed one state per edge leaving the origin. Pulling away from the kerb
    // is not a movement *through* an intersection, so none of them pays a node
    // delay or a turn penalty.
    const auto start_index = static_cast<std::size_t>(start_node);
    {
        const auto begin = static_cast<std::size_t>(adj_offsets_[start_index]);
        const auto end = static_cast<std::size_t>(adj_offsets_[start_index + 1]);
        for (std::size_t slot = begin; slot < end; ++slot) {
            const AdjEntry& entry = adjacency_[slot];
            const auto state = static_cast<std::size_t>(entry.edge_id);
            visit_stamp_[state] = generation_;
            g_score_[state] = entry.travel_time_sec;
            h_cache_[state] = heuristic(entry.target, goal.lat, goal.lon);
            came_from_state_[state] = -1;
            open.push(QueueEntry{g_score_[state] + h_cache_[state], entry.edge_id});
        }
    }

    std::uint64_t expanded = 0;
    while (!open.empty()) {
        const QueueEntry current = open.top();
        open.pop();
        const auto state = static_cast<std::size_t>(current.edge);

        // Lazy deletion: a state can sit in the heap several times, once per
        // improvement. Anything worse than its current best is a stale copy.
        if (current.f_score > g_score_[state] + h_cache_[state]) {
            continue;
        }
        // Goal test on pop, not on push. The heuristic is consistent -- a
        // global speed bound satisfies the triangle inequality, and the
        // intersection costs only ever add to an arc -- so the first pop of a
        // state sitting at the goal is already optimal.
        const Edge& arrival = edges_[state];
        if (arrival.dst == end_node) {
            return reconstruct(start_node, current.edge, expanded, preferences);
        }
        ++expanded;

        const auto node = static_cast<std::size_t>(arrival.dst);
        const auto begin = static_cast<std::size_t>(adj_offsets_[node]);
        const auto end = static_cast<std::size_t>(adj_offsets_[node + 1]);
        const float base = g_score_[state];
        // The movements leaving this state are contiguous in turn_flags_, in
        // the same order as the adjacency list being walked, so this is a
        // second sequential read rather than a lookup.
        const std::int32_t flag_base =
            weigh_flags ? turn_offsets_[state] - adj_offsets_[node] : 0;

        for (std::size_t slot = begin; slot < end; ++slot) {
            const AdjEntry& entry = adjacency_[slot];
            const auto next = static_cast<std::size_t>(entry.edge_id);
            const float preference_penalty =
                weigh_flags
                    ? penalties.by_flags[turn_flags_[static_cast<std::size_t>(
                                             flag_base + static_cast<std::int32_t>(slot))] &
                                         turn_flag::kAll]
                    : 0.0f;
            const float tentative =
                base + entry.travel_time_sec +
                transition_cost_unchecked(current.edge, entry.edge_id) +
                preference_penalty;

            if (visit_stamp_[next] == generation_ &&
                tentative >= g_score_[next]) {
                continue;
            }
            if (visit_stamp_[next] != generation_) {
                visit_stamp_[next] = generation_;
                h_cache_[next] = heuristic(entry.target, goal.lat, goal.lon);
            }
            g_score_[next] = tentative;
            came_from_state_[next] = current.edge;
            open.push(QueueEntry{tentative + h_cache_[next], entry.edge_id});
        }
    }

    result.nodes_expanded = expanded;
    return result;  // found == false
}

}  // namespace irvine

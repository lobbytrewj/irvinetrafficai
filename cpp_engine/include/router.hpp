#pragma once

// Native routing core for irvine-traffic-ai.
//
// Layout
// ------
// Road segments are graph *edges*; intersections are *nodes*. Both live in
// flat std::vectors indexed by their own id, so an id is an array offset and
// never a hash lookup.
//
// Traversal uses a CSR (compressed sparse row) adjacency: `adj_offsets_` holds
// N+1 offsets into a single `adjacency_` array, so every outgoing edge of a
// node is contiguous. At an average out-degree near 2.3, a node's whole
// adjacency list usually lands in one cache line.
//
// Dynamic weights
// ---------------
// `update_edge_weights` overwrites travel times in place, in O(E), touching no
// allocation and leaving the topology (offsets, targets, ids) untouched. The
// forecaster re-scores the network every five minutes; the topology changes
// when OpenStreetMap does, which is to say almost never.
//
// Intersection costs
// ------------------
// Rolling distance over posted speed limits describes a road with no other
// traffic on it and no signals, which is not a road anyone drives: an 8 km
// arterial trip across Irvine crosses a dozen signalised junctions, and the
// time spent stopped at them is the same order as the time spent moving.
//
// So the search charges three things beyond edge travel time, all optional and
// all zero unless configured (see set_node_delays / set_turn_penalties):
//
//   priority delay  every movement through a node pays it -- a signal, which
//                   stops cross traffic and through traffic alike.
//   yield delay     only movements that touch a minor road pay it -- a stop
//                   sign, which does not slow the arterial it sits on.
//   turn penalty    the transition itself: a left across oncoming traffic
//                   costs more than a right, which costs more than going
//                   straight.
//
// A turn penalty is a property of an edge *pair*, not of a node, so the search
// is over directed edges rather than nodes: a state is "arrived at v along
// edge e", and the cost of leaving it for e' is
//
//     travel(e') + priority(v) + [minor(e) or minor(e')] * yield(v)
//                + turn(e -> e')
//
// The origin therefore pays no intersection cost (no edge precedes it) and
// neither does the destination (none follows), which is what a driver
// experiences. The state space is E rather than N -- 15,553 rather than 6,864
// here -- and the search is otherwise unchanged.
//
// Threading
// ---------
// A Graph owns its search scratch space, so a single instance serialises
// queries on an internal mutex. The bindings release the GIL around the
// search, so other Python work proceeds during it -- but two threads sharing
// one Graph will queue. For real parallelism give each worker its own Graph,
// or run multiple processes.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace irvine {

// Mean Earth radius (IUGG). Matches src/data/graph_builder.py so C++ and
// Python heuristics agree to the metre.
inline constexpr double kEarthRadiusMeters = 6371008.8;

struct Node {
    int id = -1;
    double lat = 0.0;
    double lon = 0.0;
};

struct Edge {
    int id = -1;
    int src = -1;
    int dst = -1;
    float length_meters = 0.0f;
    float travel_time_sec = 0.0f;
};

// Functional road class, coarse on purpose: the only distinction the cost
// model makes is whether a movement touches a minor road, which is what
// decides who yields to whom at an uncontrolled junction.
enum class RoadClass : std::uint8_t {
    Freeway = 0,   // motorway/trunk and their links: grade separated
    Arterial = 1,  // primary/secondary: the signalised grid
    Minor = 2,     // tertiary and below: collectors, residential, service
};

// Per-movement hazard flags, packed one byte per transition.
//
// Deliberately *not* fields on Edge or AdjEntry, for two reasons.
//
// Correctness first: a turn is a property of an edge pair. The same segment is
// entered as a left from one approach and as a straight-on from another -- on
// the Irvine network, 94% of the edges reachable by an unprotected left are
// also reachable without one -- so a flag on the edge would charge the penalty
// to traffic going straight through.
//
// And cache behaviour, which is the reason to care where they live at all:
// AdjEntry is exactly 12 bytes, so five of them share a 64-byte line in the
// A* inner loop. Adding a byte pads it to 16 and drops that to four, costing
// ~33% more memory bandwidth on the hottest read in the engine. The flags
// instead live in their own array, indexed so that the movements leaving one
// edge are contiguous -- a second sequential stream rather than a fatter one.
namespace turn_flag {
inline constexpr std::uint8_t kNone = 0u;
inline constexpr std::uint8_t kUnprotectedLeft = 1u << 0;
inline constexpr std::uint8_t kAggressiveMerge = 1u << 1;
inline constexpr std::uint8_t kAll = kUnprotectedLeft | kAggressiveMerge;
}  // namespace turn_flag

// What a caller is willing to trade time for.
//
// These bias the search without changing the clock: a route that avoids an
// unprotected left really is slower, and RouteResult reports the time it
// actually takes. The penalty is what the driver would pay to skip the turn,
// not what the turn costs, and it is reported separately as
// RouteResult::preference_penalty_sec.
struct RoutePreferences {
    bool avoid_unprotected_lefts = false;
    bool low_stress_merges = false;
    float unprotected_left_penalty_sec = 45.0f;
    float aggressive_merge_penalty_sec = 60.0f;

    // True when nothing is switched on, which lets the search skip the lookup
    // entirely on the default path.
    bool any() const noexcept {
        return avoid_unprotected_lefts || low_stress_merges;
    }
};

// Seconds added to a movement by the turn itself, on top of any node delay.
// Signs follow compass bearings: a *left* turn in right-hand traffic crosses
// oncoming traffic and is the expensive one.
struct TurnPenalties {
    float through_sec = 0.0f;
    float right_sec = 0.0f;
    float left_sec = 0.0f;
    float uturn_sec = 0.0f;
    // A heading change within this many degrees is "straight on".
    float through_band_deg = 35.0f;
    // Beyond this it is a U-turn rather than a hard left or right.
    float uturn_band_deg = 150.0f;
};

struct RouteResult {
    // False when no path exists. Distinguishes "unreachable" from the
    // zero-length result of routing a node to itself.
    bool found = false;
    // Time on the road: travel plus node delays plus turn penalties. Never
    // includes preference penalties, so an avoidance route reports the
    // duration a driver would actually experience on it.
    double total_duration_sec = 0.0;
    double total_distance_meters = 0.0;
    // How much preference bias the winning path still carries -- zero when the
    // search successfully avoided everything it was asked to. Non-zero means
    // the alternatives were worse, which is worth being able to see.
    double preference_penalty_sec = 0.0;
    std::vector<int> path_node_ids;
    std::vector<int> path_edge_ids;
    // Nodes popped from the open set. Reported because it, not wall-clock
    // time, is what says whether the heuristic is doing its job.
    std::uint64_t nodes_expanded = 0;
};

// Great-circle distance in metres.
double haversine_meters(double lat1, double lon1, double lat2, double lon2) noexcept;

// Engine version string.
const char* version() noexcept;

class Graph {
public:
    explicit Graph(std::size_t num_nodes);

    // -- construction -------------------------------------------------------
    // `id` indexes directly into the node array and must be in [0, num_nodes).
    void add_node(int id, double lat, double lon);

    // `edge_id` indexes directly into the edge array. Ids must be dense over
    // [0, E); gaps are rejected by finalize().
    void add_edge(int edge_id, int u, int v, float length_meters,
                  float travel_time_sec);

    // Build the CSR adjacency. Called automatically by the first query or
    // weight update; exposed so a benchmark can pay the cost up front instead
    // of charging it to the first query. Idempotent.
    void finalize();

    // -- dynamic weights ----------------------------------------------------
    // Overwrite every edge's travel time, in edge-id order. O(E), no
    // allocation, topology untouched. Size must equal num_edges().
    void update_edge_weights(const std::vector<float>& new_travel_times);

    // Overwrite a subset, by edge id. O(k) for k updates.
    //
    // The forecaster covers 444 corridor segments out of 15,553 edges, so the
    // full-vector form above makes the caller build and copy a mostly-unchanged
    // 15,553-element array every tick. This is the same operation restricted to
    // the edges that actually moved.
    void update_edge_weights_subset(const std::vector<int>& edge_ids,
                                    const std::vector<float>& new_travel_times);

    // -- intersection costs -------------------------------------------------
    // Per-node delay in seconds, indexed by node id. `priority_sec` is charged
    // to every movement through the node; `yield_sec` only to movements that
    // enter or leave a Minor-class edge. Pass empty vectors to clear both.
    //
    // A signalised arterial crossing is (priority=30, yield=0): the light
    // stops everyone. An unsignalised residential junction is (0, 10): the
    // arterial through movement keeps its right of way and pays nothing, the
    // driver turning off it does not.
    void set_node_delays(const std::vector<float>& priority_sec,
                         const std::vector<float>& yield_sec);

    // Road class per edge id, as RoadClass values. Only the yield rule reads
    // it. Empty clears it, which makes every yield delay inapplicable.
    void set_edge_road_classes(const std::vector<std::uint8_t>& classes);

    void set_turn_penalties(const TurnPenalties& penalties);
    const TurnPenalties& turn_penalties() const noexcept { return turns_; }

    // Compass bearing in degrees at each edge's start and end, for the turn
    // angle. Optional: finalize() derives both from the straight src -> dst
    // chord when they are not supplied, which is exact for a straight segment
    // and approximate for a curved one. Pass the real geometry's first and
    // last bearings to remove that approximation.
    void set_edge_bearings(const std::vector<float>& entry_deg,
                           const std::vector<float>& exit_deg);

    // True once any delay or turn penalty is configured.
    bool has_intersection_costs() const noexcept { return intersection_costs_; }

    // Per-movement hazard flags, as a sparse (in_edge, out_edge) -> flags
    // table; anything omitted is turn_flag::kNone. This is exactly the shape
    // src/data/graph_builder.py's TurnTable exports.
    //
    // Finalizes the graph if it is not already, since the packed layout is
    // indexed off the CSR adjacency -- so supply edge bearings first, or
    // finalize() will derive them from chords. Adding a node or an edge
    // afterwards clears the table, because the indices it was built against no
    // longer describe the graph.
    void set_turn_flags(const std::vector<int>& in_edge_ids,
                        const std::vector<int>& out_edge_ids,
                        const std::vector<std::uint8_t>& flags);

    // Flags for one movement. Throws if the edges do not meet head to tail.
    std::uint8_t turn_flags(int in_edge, int out_edge) const;

    bool has_turn_flags() const noexcept { return !turn_flags_.empty(); }

    // Seconds added at the node shared by two consecutive edges: node delays
    // plus the turn penalty, plus any preference penalty the flags trigger.
    // Throws std::invalid_argument if the edges do not meet head to tail.
    double transition_cost(int in_edge, int out_edge,
                           const RoutePreferences& preferences = {}) const;

    // Total seconds for a path given as consecutive edge ids: every edge's
    // travel time plus every interior transition cost. This is what
    // find_shortest_path minimises, exposed so a caller can score a path it
    // did not search for -- a route from another graph, say -- under this
    // graph's weights and delays. Pass the same preferences the search used to
    // get the same number back.
    double score_path(const std::vector<int>& edge_ids,
                      const RoutePreferences& preferences = {}) const;

    // Seconds of preference bias a path carries: the part of its search cost
    // that is not time on the road.
    double preference_penalty(const std::vector<int>& edge_ids,
                              const RoutePreferences& preferences) const;

    // -- query --------------------------------------------------------------
    // A* over travel time plus intersection costs, with a Haversine lower
    // bound as the heuristic. Throws std::out_of_range on an unknown node id.
    //
    // `preferences` adds a penalty to any movement carrying a flag it asks to
    // avoid. The penalties are non-negative, so the heuristic stays admissible
    // and the result stays optimal -- under the preferred cost, which is the
    // point: the route returned is the one the driver would choose, not the
    // fastest one.
    RouteResult find_shortest_path(int start_node, int end_node,
                                   const RoutePreferences& preferences = {});

    // -- accessors ----------------------------------------------------------
    std::size_t num_nodes() const noexcept { return nodes_.size(); }
    std::size_t num_edges() const noexcept { return edges_.size(); }
    bool is_finalized() const noexcept { return finalized_; }

    // Fastest edge in the graph, in m/s. The heuristic divides by this to turn
    // metres into a lower bound on seconds; see find_shortest_path.
    double max_speed_mps() const noexcept { return max_speed_mps_; }

    const Node& node(int id) const;
    const Edge& edge(int id) const;

private:
    // One outgoing edge, in CSR order. 12 bytes, so ~5 fit a cache line and
    // the inner loop of A* reads a single sequential stream.
    //
    // travel_time_sec duplicates Edge::travel_time_sec. That copy is a derived
    // cache of the canonical value in edges_, and every write to either goes
    // through write_travel_time(), so the two cannot drift apart.
    struct AdjEntry {
        int target;
        int edge_id;
        float travel_time_sec;
    };

    // One search state: "arrived at edge.dst along edge". The heap is keyed by
    // f = g + h as before; only what a state *is* has changed.
    struct QueueEntry {
        float f_score;
        int edge;
        // std::priority_queue is a max-heap; std::greater over this inverts it.
        bool operator>(const QueueEntry& other) const noexcept {
            return f_score > other.f_score;
        }
    };

    void ensure_finalized();
    void build_turn_flag_index();
    void clear_turn_flags() noexcept;
    // Slot of `out_edge` inside its source node's adjacency list, or -1.
    std::int32_t adjacency_slot(int out_edge) const noexcept;
    // Penalty lookup indexed by the two flag bits, built once per query.
    struct PenaltyTable {
        float by_flags[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        bool active = false;
    };
    static PenaltyTable penalty_table(const RoutePreferences& preferences) noexcept;
    void write_travel_time(int edge_id, float seconds) noexcept;
    void refresh_max_speed() noexcept;
    void refresh_intersection_costs() noexcept;
    void compute_chord_bearings();
    // Bump the visit stamp, which invalidates all per-state scratch at once
    // instead of clearing E entries per query.
    void begin_search();
    [[nodiscard]] bool visited(int edge) const noexcept {
        return visit_stamp_[static_cast<std::size_t>(edge)] == generation_;
    }
    float heuristic(int node, double goal_lat, double goal_lon) noexcept;
    // Node delays plus turn penalty, with both edges already validated.
    float transition_cost_unchecked(int in_edge, int out_edge) const noexcept;
    float turn_penalty(int in_edge, int out_edge) const noexcept;
    RouteResult reconstruct(int start_node, int goal_edge,
                            std::uint64_t expanded,
                            const RoutePreferences& preferences) const;

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<bool> node_assigned_;
    std::vector<bool> edge_assigned_;

    // CSR adjacency: adjacency_[adj_offsets_[n] .. adj_offsets_[n+1]) are the
    // outgoing edges of node n.
    std::vector<std::int32_t> adj_offsets_;
    std::vector<AdjEntry> adjacency_;
    // Edge id -> its slot in adjacency_, so a weight update can reach the CSR
    // mirror without scanning.
    std::vector<std::int32_t> edge_to_slot_;

    // Intersection cost model. Every one of these is empty or zero by default,
    // and a Graph with none of them set searches exactly the travel times it
    // was given.
    std::vector<float> node_priority_delay_;
    std::vector<float> node_yield_delay_;
    std::vector<std::uint8_t> edge_road_class_;
    std::vector<float> edge_entry_bearing_;
    std::vector<float> edge_exit_bearing_;
    TurnPenalties turns_{};
    bool intersection_costs_ = false;

    // Movement flags in transition-CSR order: the movements leaving edge e
    // occupy turn_flags_[turn_offsets_[e] .. turn_offsets_[e + 1]), in the
    // same order as the CSR adjacency of e's destination node. One byte per
    // movement -- 40,572 of them for Irvine, so the whole table is 40 KB and
    // the offsets 62 KB.
    std::vector<std::int32_t> turn_offsets_;
    std::vector<std::uint8_t> turn_flags_;
    // Kept so the packed layout can be rebuilt after a topology change.
    std::vector<int> pending_turn_in_;
    std::vector<int> pending_turn_out_;
    std::vector<std::uint8_t> pending_turn_flags_;

    // Search scratch, sized once at finalize() and reused across queries. One
    // entry per directed edge, since a state is an edge and not a node.
    std::vector<float> g_score_;
    std::vector<float> h_cache_;
    std::vector<std::int32_t> came_from_state_;
    std::vector<std::uint32_t> visit_stamp_;
    std::uint32_t generation_ = 0;

    double max_speed_mps_ = 0.0;
    bool finalized_ = false;
    mutable std::mutex search_mutex_;
};

}  // namespace irvine

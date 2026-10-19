#pragma once
#include "solver/Solver.h"
#include "vk/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace swe {

namespace vkc { class Context; }
struct CaseData;

// Records h, qx, qy over time into one `.swerec` file with the CPU doing as little as possible:
//  - the solver packs the state to 16 bits per cell on the GPU and copies it, with a wet flag per 16 x 16 tile
//    and the scalars (exact t, step), into one of kSlots host-cached buffers inside the frame's command buffer:
//    no extra submission, no wait; the frame's fence tells when the slot is readable;
//  - a writer thread then gathers the wet tiles only (6 bytes per cell), releases the slot, and appends the
//    record to the file; the index (t, step, offset per record) is written at the end when recording stops.
// Snapshots are taken every `every` seconds of simulated time (the first at the start). When every slot is still
// busy the snapshot waits for the next frame (counted in Stats::delayed).
//
// File layout (little endian): Header (256 bytes), records, then Header::count IndexEntry. A record is
// RecordHeader, the solver scalars of that state (Header::scalarWords uint32, version 2: exact clock, volumes
// through the boundaries and the zones, see solver/SolverParams.h ScalarSlot), uint32 tile ids
// (ti * tilesY + tj), then per tile 256 cells (di major, dj minor), each the
// solver's (h, hu, hv) as three float32 (encoding 2, the default: bit-exact), or packed to 16 bits (encoding 1:
// uint16 h in mm, float16 hu, float16 hv). Cells outside the grid (edge tiles) are zero. tools/swerec.py reads it.
class Recorder
{
  public:
	static constexpr uint32_t kSlots = 3;

	struct Header
	{
		char     magic[8] = {'S', 'W', 'E', 'R', 'E', 'C', '1', 0};
		uint32_t version  = 2;        // 2: each record carries the solver scalars (scalarWords words)
		uint32_t nx = 0, ny = 0, tile = 16, tilesX = 0, tilesY = 0;
		uint32_t encoding = 2;        // 2: h, hu, hv float32; 1: h uint16 mm, hu / hv float16
		uint32_t pad0     = 0;
		double   dx = 0, dy = 0, z_ref = 0;
		uint64_t indexOffset = 0;        // patched when the file is closed
		uint32_t count       = 0;        //   "
		uint32_t scalarWords = 0;        // version 2: uint32 words of solver scalars after each RecordHeader
		char     caseName[64] = {};
		char     reserved[256 - 8 - 4 * 8 - 3 * 8 - 8 - 8 - 64] = {};
	};
	static_assert(sizeof(Header) == 256);

	struct RecordHeader
	{
		double   t     = 0;
		uint32_t step  = 0;
		uint32_t tiles = 0;        // wet tiles in the record
	};
	struct IndexEntry
	{
		uint64_t offset = 0;        // of the RecordHeader
		double   t      = 0;
		uint32_t step   = 0;
		uint32_t tiles  = 0;
	};

	struct Stats
	{
		size_t   records = 0, delayed = 0, pending = 0;
		uint64_t bytes   = 0;
		double   lastT   = 0.0;
	};

	Recorder(vkc::Context const &ctx, Solver &solver);
	~Recorder();

	// Opens the file and allocates the slots for the loaded case; `every` in simulated seconds; `halfPrecision`
	// packs the cells to 16 bits (half the bytes, h to the mm, hu / hv to 3 significant digits)
	bool start(std::filesystem::path const &path, CaseData const &data, double every, bool halfPrecision = false);
	bool halfPrecision() const { return half_; }
	// Flushes the writer, writes the index and closes the file (also called by the destructor)
	void stop();
	bool active() const { return active_; }
	std::filesystem::path const &path() const { return path_; }
	double every() const { return every_; }
	double nextTime() const { return nextT_; }        // simulated time of the next snapshot wanted

	// Frame loop. due(t): a snapshot is wanted for the state at simulated time t (the last known one).
	// acquire(frameSlot) returns a free slot marked as filled by that frame, if any; the caller records
	// Solver::recordSnapshot into slotBuffer(slot). collect(frameSlot) hands the slots filled by that frame to
	// the writer: call it once the frame's fence has signalled.
	bool                    due(double t) const;
	std::optional<uint32_t> acquire(uint32_t frameSlot, double t);
	vk::Buffer              slotBuffer(uint32_t slot) const { return slots_[slot].buffer.handle(); }
	void                    collect(uint32_t frameSlot);

	// Synchronous snapshot of the current state (batch runs, between two batches of steps): packs, copies,
	// waits for the device and hands the slot to the writer. `t` is the simulated time of that state.
	bool snapshotNow(double t);

	Stats stats() const;

  private:
	enum class State : int
	{
		Free,
		Filling,        // recorded in a frame whose fence is pending
		Queued,         // readable, waiting for the writer
	};
	struct Slot
	{
		vkc::Buffer       buffer;
		std::atomic<int>  state{int(State::Free)};
		uint32_t          frame = 0;
	};

	void writerLoop();
	void writeRecord(Slot &slot);

	vkc::Context const        &ctx_;
	Solver                    &solver_;
	std::array<Slot, kSlots>   slots_;
	bool                       active_ = false;
	std::filesystem::path      path_;
	double                     every_ = 1.0, nextT_ = 0.0;
	Header                     header_;
	uint32_t                   nx_ = 0, ny_ = 0, tilesX_ = 0, tilesY_ = 0;
	size_t                     cells_ = 0, tiles_ = 0;
	bool                       half_ = false;

	// writer thread
	std::thread                writer_;
	std::mutex                 mutex_;
	std::condition_variable    cv_;
	std::deque<uint32_t>       queue_;
	bool                       quit_ = false;
	FILE                      *file_ = nullptr;
	std::vector<IndexEntry>    index_;
	std::vector<uint8_t>       gather_;        // writer's own copy of the wet tiles
	std::vector<uint32_t>      scalarCopy_;    //   ... and of the scalars
	// stats (written by the writer, read by the GUI)
	std::atomic<size_t>        records_{0}, delayed_{0};
	std::atomic<uint64_t>      bytes_{0};
	std::atomic<double>        lastT_{0.0};
};

}        // namespace swe

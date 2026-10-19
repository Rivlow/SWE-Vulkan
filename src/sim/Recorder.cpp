#include "sim/Recorder.h"

#include "cases/Case.h"
#include "util/Log.h"
#include "vk/Context.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace swe {

namespace {

constexpr uint32_t kTile = 16;

}        // namespace

Recorder::Recorder(vkc::Context const &ctx, Solver &solver) :
    ctx_(ctx), solver_(solver)
{
}

Recorder::~Recorder()
{
	stop();
}

bool Recorder::start(std::filesystem::path const &path, CaseData const &data, double every, bool halfPrecision)
{
	stop();
	if (!solver_.tileCount())
		return false;
	std::filesystem::create_directories(path.parent_path());
	file_ = std::fopen(path.string().c_str(), "wb");
	if (!file_)
	{
		log::warn("recorder: cannot open {}", path.string());
		return false;
	}
	std::setvbuf(file_, nullptr, _IOFBF, size_t(4) << 20);
	path_   = path;
	every_  = std::max(every, 0.0);
	nextT_  = 0.0;
	nx_     = data.nx;
	ny_     = data.ny;
	tilesX_ = solver_.tilesX();
	tilesY_ = solver_.tilesY();
	cells_  = data.cells();
	tiles_  = size_t(tilesX_) * tilesY_;
	half_   = halfPrecision;
	header_ = Header{};
	header_.encoding = half_ ? 1 : 2;
	header_.nx     = nx_;
	header_.ny     = ny_;
	header_.tilesX = tilesX_;
	header_.tilesY = tilesY_;
	header_.dx     = data.dx();
	header_.dy     = data.dy();
	header_.z_ref  = data.z_ref;
	header_.scalarWords = S_COUNT;
	std::snprintf(header_.caseName, sizeof(header_.caseName), "%s", data.name.c_str());
	std::fwrite(&header_, sizeof(header_), 1, file_);
	index_.clear();
	records_ = delayed_ = 0;
	bytes_   = sizeof(header_);
	lastT_   = 0.0;

	solver_.enableRecording(true, half_);
	for (uint32_t s = 0; s < kSlots; ++s)
	{
		slots_[s].buffer = vkc::Buffer::readback(ctx_, solver_.snapshotBytes(), std::format("record-slot-{}", s));
		slots_[s].state  = int(State::Free);
	}
	quit_   = false;
	writer_ = std::thread([this] { writerLoop(); });
	active_ = true;
	log::info("recorder: {} ({} x {} cells, every {:.3g} s, {})", path.string(), nx_, ny_, every_, half_ ? "16-bit" : "float32");
	return true;
}

void Recorder::stop()
{
	if (!active_)
		return;
	active_ = false;
	// slots still in flight belong to frames the device may be executing: let them finish, the writer takes them
	ctx_.device.waitIdle();
	for (uint32_t s = 0; s < kSlots; ++s)
		if (slots_[s].state == int(State::Filling))
		{
			slots_[s].state = int(State::Queued);
			std::lock_guard lock(mutex_);
			queue_.push_back(s);
		}
	{
		std::lock_guard lock(mutex_);
		quit_ = true;
	}
	cv_.notify_all();
	if (writer_.joinable())
		writer_.join();
	// index and patched header
	if (file_)
	{
		uint64_t indexOffset = static_cast<uint64_t>(std::ftell(file_));
		std::fwrite(index_.data(), sizeof(IndexEntry), index_.size(), file_);
		header_.indexOffset = indexOffset;
		header_.count       = static_cast<uint32_t>(index_.size());
		std::fseek(file_, 0, SEEK_SET);
		std::fwrite(&header_, sizeof(header_), 1, file_);
		std::fclose(file_);
		file_ = nullptr;
		bytes_ += index_.size() * sizeof(IndexEntry);
		log::info("recorder: {} records, {:.1f} MB in {}", index_.size(), double(bytes_.load()) / 1048576.0, path_.string());
	}
	solver_.enableRecording(false);
	for (auto &s : slots_)
		s.buffer = {};
}

bool Recorder::due(double t) const
{
	return active_ && std::isfinite(t) && t >= nextT_;
}

std::optional<uint32_t> Recorder::acquire(uint32_t frameSlot, double t)
{
	if (!active_)
		return std::nullopt;
	for (uint32_t s = 0; s < kSlots; ++s)
	{
		int expected = int(State::Free);
		if (slots_[s].state.compare_exchange_strong(expected, int(State::Filling)))
		{
			slots_[s].frame = frameSlot;
			// next snapshot: the first multiple of `every` beyond this one's time
			nextT_ = every_ > 0.0 ? (std::floor(t / every_) + 1.0) * every_ : t;
			return s;
		}
	}
	++delayed_;
	return std::nullopt;
}

void Recorder::collect(uint32_t frameSlot)
{
	if (!active_)
		return;
	bool any = false;
	for (uint32_t s = 0; s < kSlots; ++s)
		if (slots_[s].state == int(State::Filling) && slots_[s].frame == frameSlot)
		{
			slots_[s].state = int(State::Queued);
			std::lock_guard lock(mutex_);
			queue_.push_back(s);
			any = true;
		}
	if (any)
		cv_.notify_one();
}

bool Recorder::snapshotNow(double t)
{
	constexpr uint32_t kSyncFrame = ~0u;        // frame tag of the synchronous path
	auto               slot       = acquire(kSyncFrame, t);
	if (!slot)
		return false;
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) { solver_.recordSnapshot(cb, slotBuffer(*slot)); });
	collect(kSyncFrame);
	return true;
}

Recorder::Stats Recorder::stats() const
{
	Stats st;
	st.records = records_;
	st.delayed = delayed_;
	st.bytes   = bytes_;
	st.lastT   = lastT_;
	for (auto const &s : slots_)
		if (s.state != int(State::Free))
			++st.pending;
	return st;
}

void Recorder::writerLoop()
{
	for (;;)
	{
		uint32_t slot;
		{
			std::unique_lock lock(mutex_);
			cv_.wait(lock, [&] { return quit_ || !queue_.empty(); });
			if (queue_.empty())
				return;        // quit, nothing left
			slot = queue_.front();
			queue_.pop_front();
		}
		writeRecord(slots_[slot]);
	}
}

// Gathers the wet tiles of the slot into the writer's buffer (the slot is free again), then appends the record
void Recorder::writeRecord(Slot &slot)
{
	size_t const wordsPerCell = half_ ? 2 : 3;
	size_t const cellBytes    = half_ ? 6 : 12;        // 16-bit: uint16 h (mm), float16 hu, hv; else three floats
	auto const  *words        = static_cast<uint32_t const *>(slot.buffer.mapped());
	auto const  *flags        = words + wordsPerCell * cells_;
	auto const  *scalars      = flags + tiles_;
	double       t            = std::ldexp(double(scalars[S_TIME_HI]), -8) + std::ldexp(double(scalars[S_TIME_LO]), -40);
	uint32_t     step         = scalars[S_NSTEPS];

	scalarCopy_.assign(scalars, scalars + S_COUNT);
	std::vector<uint32_t> ids;
	for (uint32_t tl = 0; tl < tiles_; ++tl)
		if (flags[tl])
			ids.push_back(tl);
	gather_.resize(ids.size() * kTile * kTile * cellBytes);
	uint8_t *out = gather_.data();
	for (uint32_t tl : ids)
	{
		uint32_t ti = tl / tilesY_, tj = tl % tilesY_;
		for (uint32_t di = 0; di < kTile; ++di)
		{
			uint32_t i = ti * kTile + di;
			// interior tiles: a whole row of 16 cells is contiguous in the snapshot
			if (i < nx_ && tj * kTile + kTile <= ny_)
			{
				size_t k = size_t(i) * ny_ + tj * kTile;
				if (half_)
					for (uint32_t dj = 0; dj < kTile; ++dj, out += cellBytes)
					{
						uint32_t w0 = words[2 * (k + dj)], w1 = words[2 * (k + dj) + 1];
						uint16_t v[3] = {uint16_t(w0 & 0xffffu), uint16_t(w0 >> 16), uint16_t(w1 & 0xffffu)};
						std::memcpy(out, v, cellBytes);
					}
				else
				{
					std::memcpy(out, words + 3 * k, kTile * cellBytes);
					out += kTile * cellBytes;
				}
				continue;
			}
			for (uint32_t dj = 0; dj < kTile; ++dj, out += cellBytes)
			{
				uint32_t j = tj * kTile + dj;
				if (i >= nx_ || j >= ny_)
				{
					std::memset(out, 0, cellBytes);
					continue;
				}
				size_t k = size_t(i) * ny_ + j;
				if (half_)
				{
					uint32_t w0 = words[2 * k], w1 = words[2 * k + 1];
					uint16_t v[3] = {uint16_t(w0 & 0xffffu), uint16_t(w0 >> 16), uint16_t(w1 & 0xffffu)};
					std::memcpy(out, v, cellBytes);
				}
				else
					std::memcpy(out, words + 3 * k, cellBytes);
			}
		}
	}
	slot.state = int(State::Free);        // the GPU may refill it while we write

	RecordHeader rh{.t = t, .step = step, .tiles = static_cast<uint32_t>(ids.size())};
	uint64_t     offset = static_cast<uint64_t>(std::ftell(file_));
	std::fwrite(&rh, sizeof(rh), 1, file_);
	std::fwrite(scalarCopy_.data(), sizeof(uint32_t), scalarCopy_.size(), file_);
	std::fwrite(ids.data(), sizeof(uint32_t), ids.size(), file_);
	std::fwrite(gather_.data(), 1, gather_.size(), file_);
	index_.push_back({.offset = offset, .t = t, .step = step, .tiles = rh.tiles});
	bytes_ += sizeof(rh) + scalarCopy_.size() * sizeof(uint32_t) + ids.size() * sizeof(uint32_t) + gather_.size();
	++records_;
	lastT_ = t;
}

}        // namespace swe

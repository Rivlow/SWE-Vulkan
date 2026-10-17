#pragma once
#include <atomic>
#include <mutex>
#include <string>

namespace swe::geo {

// Shared between a background job and the GUI thread.
class Progress
{
  public:
	void set(float fraction, std::string stage)
	{
		fraction_.store(fraction);
		std::lock_guard lock(mutex_);
		stage_ = std::move(stage);
	}
	void  setFraction(float f) { fraction_.store(f); }
	float fraction() const { return fraction_.load(); }
	std::string stage() const
	{
		std::lock_guard lock(mutex_);
		return stage_;
	}
	void cancel() { cancelled_.store(true); }
	bool cancelled() const { return cancelled_.load(); }
	void reset()
	{
		fraction_.store(0.0f);
		cancelled_.store(false);
		std::lock_guard lock(mutex_);
		stage_.clear();
	}

  private:
	std::atomic<float>  fraction_{0.0f};
	std::atomic<bool>   cancelled_{false};
	mutable std::mutex  mutex_;
	std::string         stage_;
};

}        // namespace swe::geo
